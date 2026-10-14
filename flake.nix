{
  description = "A framework for automatic performance optimization of DNN training and inference";

  nixConfig = {
    bash-prompt-prefix = "(ff) ";
    extra-substituters = [
      "https://ff.cachix.org"
      #"https://cuda-maintainers.cachix.org/"
    ];
    extra-trusted-public-keys = [
      #"cuda-maintainers.cachix.org-1:0dq3bujKpuEPMCX6U4WylrUDZ9JyUG0VpVZa7CNfq5E="
      "ff.cachix.org-1:IRdsNEnht4YKGUasP6SX5DfpaOTBckhpJDEODz7wMFM="
    ];
  };

  inputs = {
    nixpkgs.url = "nixpkgs/nixos-23.11";
    flake-utils.url = "github:numtide/flake-utils";

    proj-repo = {
      url = "git+https://git.sr.ht/~lockshaw/proj";
      inputs.nixpkgs.follows = "nixpkgs";
      inputs.flake-utils.follows = "flake-utils";
    };

    nixGL = {
      url = "github:nix-community/nixGL";
      inputs.nixpkgs.follows = "nixpkgs";
      inputs.flake-utils.follows = "flake-utils";
    };
  };

  outputs = { self, nixpkgs, flake-utils, proj-repo, nixGL, ... }: flake-utils.lib.eachSystem [ "x86_64-linux" ] (system:
    let
      pkgs = import nixpkgs {
        inherit system;
        config.allowUnfree = true;
      };
      lib = pkgs.lib;

      mkShell = attrs: pkgs.mkShell.override {
        stdenv = pkgs.cudaPackages.backendStdenv;
      } (attrs // {
        hardeningDisable = ["all"]; # disable nixpkgs default compiler arguments, otherwise ubsan doesn't catch
                                    # signed overflows due to the signedoverflow hardening setting.
                                    # for more details, see the following (long-running) nixpkgs github issues:
                                    # - https://github.com/NixOS/nixpkgs/issues/18995
                                    # - https://github.com/NixOS/nixpkgs/issues/60919
      });

      proj = proj-repo.packages.${system}.proj;
    in
    {
      packages = rec {
        libdwarf-lite = pkgs.callPackage ./.flake/pkgs/libdwarf-lite.nix { };
        cpptrace = pkgs.callPackage ./.flake/pkgs/cpptrace.nix { inherit libdwarf-lite; };
        libassert = pkgs.callPackage ./.flake/pkgs/libassert.nix { inherit cpptrace; };
        realm = pkgs.callPackage ./.flake/pkgs/realm.nix { };
        bencher-cli = pkgs.callPackage ./.flake/pkgs/bencher-cli.nix { };
        ffdb = pkgs.callPackage ./.flake/pkgs/ffdb { inherit proj; };
        hpp2plantuml = pkgs.python3Packages.callPackage ./.flake/pkgs/hpp2plantuml.nix { };
        fccf = pkgs.callPackage ./.flake/pkgs/fccf { };
        rapidcheckFull = pkgs.symlinkJoin {
          name = "rapidcheckFull";
          paths = (with pkgs; [ rapidcheck.out rapidcheck.dev ]);
        };
        doctest = pkgs.doctest.overrideAttrs ( old: rec {
          version = "2.4.9";
          src = pkgs.fetchFromGitHub {
            owner = "doctest";
            repo = "doctest";
            rev = "v${version}";
            sha256 = "sha256-ugmkeX2PN4xzxAZpWgswl4zd2u125Q/ADSKzqTfnd94=";
          };
          patches = [
            ./.flake/patches/doctest-template-test.patch
          ];
        });
      };

      devShells = rec {
        ci = mkShell {
          shellHook = ''
            export RC_PARAMS="max_discard_ratio=100"
          '';

          buildInputs = builtins.concatLists [
            (with pkgs; [
              zlib
              boost
              nlohmann_json
              spdlog
              range-v3
              fmt
              cmakeCurses
              ccache
              pkg-config
              python3
              cudatoolkit
              cudaPackages.cuda_nvcc
              cudaPackages.cudnn
              cudaPackages.nccl
              cudaPackages.libcublas
              cudaPackages.cuda_cudart
              tl-expected
              doxygen
              lcov # for code coverage
              compdb
              gbenchmark
              libtorch-bin
              graphviz # for documentation
              texliveBasic # for documentation
            ])
            (with proj-repo.packages.${system}; [
              proj
            ])
            (with self.packages.${system}; [
              libassert
              realm
              rapidcheckFull
              doctest
            ])
          ];
        };

        gpu-ci = mkShell {
          inputsFrom = [ ci ];
          hardeningDisable = [ "all" ];

          buildInputs = builtins.concatLists [
            (with nixGL.packages.${system}; [
              nixGLDefault
            ])
          ];
        };

        default = mkShell {
          inputsFrom = [ ci ];

          VIMPLUGINS = lib.strings.concatStringsSep "," [
            "${proj-repo.packages.${system}.proj-nvim}"
          ];

          hardeningDisable = [ "all" ];

          buildInputs = builtins.concatLists [
            (with pkgs; [
              clang-tools
              gh-markdown-preview
              shellcheck
              plantuml
              ruff
              jq
              gh
              expect
              universal-ctags
              ninja
            ])
            (with pkgs.python3Packages; [
              gitpython
              ipython
              mypy
              python-lsp-server
              pylsp-mypy
              python-lsp-ruff
              pygithub
              sqlitedict
              frozendict
              black
              toml
              numpy
            ])
            (with self.packages.${system}; [
              ffdb
              hpp2plantuml
              fccf
            ])
          ];
        };

        gpu = mkShell {
          inputsFrom = [ gpu-ci default ];
        };
      };
    }
  );
}

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
    nixpkgs.url = "nixpkgs/nixos-26.05";
    flake-utils.url = "github:numtide/flake-utils";

    proj-repo = {
      url = "git+https://github.com/elliottslaughter/proj.git?ref=refs/heads/update-nix&rev=268056035befca5f811f21fc86811498694d6254";
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

      nixgl = pkgs.callPackage ./.flake/pkgs/nixgl { inherit pkgs; src = nixGL; };
    in
    {
      packages = rec {
        libdwarf-lite = pkgs.callPackage ./.flake/pkgs/libdwarf-lite.nix { };
        cpptrace = pkgs.callPackage ./.flake/pkgs/cpptrace.nix { inherit libdwarf-lite; };
        libassert = pkgs.callPackage ./.flake/pkgs/libassert.nix { inherit cpptrace; };
        realm = pkgs.callPackage ./.flake/pkgs/realm.nix { };
        cudnn = pkgs.callPackage ./.flake/pkgs/cudnn.nix { };
        bencher-cli = pkgs.callPackage ./.flake/pkgs/bencher-cli.nix { };
        ffdb = pkgs.callPackage ./.flake/pkgs/ffdb { inherit proj; };
        robotpy-cppheaderparser = pkgs.python3Packages.callPackage ./.flake/pkgs/robotpy-cppheaderparser.nix { };
        hpp2plantuml = pkgs.python3Packages.callPackage ./.flake/pkgs/hpp2plantuml.nix { inherit robotpy-cppheaderparser; };
        fccf = pkgs.callPackage ./.flake/pkgs/fccf { };
        rapidcheckFull = pkgs.symlinkJoin {
          name = "rapidcheckFull";
          paths = (with pkgs; [ rapidcheck.out rapidcheck.dev ]);
        };
      };

      devShells = rec {
        ci = mkShell {
          shellHook = ''
            export RC_PARAMS="max_discard_ratio=100"

            # Nix passes dependency include paths through NIX_CFLAGS_COMPILE
            # rather than on the compiler command line, so ccache cannot see
            # them change. Its manifests reference store paths that are still
            # present and unmodified, so bumping a dependency (e.g. fmt) yields
            # false cache hits that return objects built against the old
            # headers. Folding the flags into the hash invalidates those.
            # -frandom-seed is a per-derivation nonce, so it is dropped to keep
            # unrelated devshell edits from invalidating the whole cache.
            # The flags are reduced to a digest and combined with "%compiler% -v"
            # so that the compiler's own identity keeps being hashed too --
            # setting a plain "string:" check would drop it, and a gcc bump that
            # left the include paths untouched would then go unnoticed.
            ccache_flag_id="$(
              printf '%s' "$NIX_CFLAGS_COMPILE" \
                | tr ' ' '\n' \
                | grep -v '^-frandom-seed=' \
                | sha256sum \
                | cut -d' ' -f1
            )"
            export CCACHE_COMPILERCHECK="%compiler% -v; echo $ccache_flag_id"
            unset ccache_flag_id

            # cudaPackages.backendStdenv pins gcc to a version cuda accepts,
            # but the wrapper still puts the default stdenv gcc's library
            # directory ahead of it, so -lgcov resolves to a libgcov whose
            # format does not match the instrumentation the pinned gcc emits.
            # Every coverage run then dies with "Version mismatch" and writes
            # no .gcda at all. This directory holds only static archives
            # (libgcc, libgcov) and crt objects, so preferring it does not
            # affect libstdc++ resolution.
            gcc_static_lib_dir="$(dirname "$(''${CXX:-g++} -print-file-name=libgcov.a)")"
            case "$gcc_static_lib_dir" in
              /*) export NIX_CFLAGS_LINK="-L$gcc_static_lib_dir $NIX_CFLAGS_LINK" ;;
            esac
            unset gcc_static_lib_dir
          '';

          buildInputs = builtins.concatLists [
            (with pkgs; [
              zlib
              boost
              nlohmann_json
              (spdlog.override { fmt = fmt_10; })
              range-v3
              fmt_10
              cmakeCurses
              ccache
              pkg-config
              python3
              cudatoolkit
              cudaPackages.cuda_nvcc
              cudaPackages.nccl
              cudaPackages.libcublas
              cudaPackages.cuda_cudart
              doctest
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
              cudnn
              libassert
              realm
              rapidcheckFull
            ])
          ];
        };

        gpu-ci = mkShell {
          inputsFrom = [ ci ];
          hardeningDisable = [ "all" ];

          buildInputs = builtins.concatLists [
            [
              nixgl.auto.nixGLDefault
            ]
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
              tig
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

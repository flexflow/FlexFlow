{
  description = "A framework for automatic performance optimization of DNN training and inference";

  nixConfig = {
    bash-prompt-prefix = "(ff) ";
    extra-substituters = [
      "https://ff.cachix.org"
      "https://cuda-maintainers.cachix.org/"
      "https://llama-cpp.cachix.org"
      "https://nixos-rocm.cachix.org/"
    ];
    extra-trusted-public-keys = [
      "cuda-maintainers.cachix.org-1:0dq3bujKpuEPMCX6U4WylrUDZ9JyUG0VpVZa7CNfq5E="
      "ff.cachix.org-1:/kyZ0w35ToSJBjpiNfPLrL3zTjuPkUiqf2WH0GIShXM="
      "nixos-rocm.cachix.org-1:VEpsf7pRIijjd8csKjFNBGzkBqOmw8H9PRmgAq14LnE="
      "llama-cpp.cachix.org-1:H75X+w83wUKTIPSO1KWy9ADUrzThyGs8P5tmAbkWhQc="
    ];
  };

  inputs = {
    nixpkgs.url = "nixpkgs/nixos-23.11";
    flake-utils.url = "github:numtide/flake-utils";

    proj-repo = {
      url = "github:lockshaw/proj";
      inputs.nixpkgs.follows = "nixpkgs";
      inputs.flake-utils.follows = "flake-utils";
    };
  };

  outputs = { self, nixpkgs, flake-utils, proj-repo, ... }: flake-utils.lib.eachSystem [ "x86_64-linux" ] (system: 
    let 
      pkgs = import nixpkgs {
        inherit system;
        config.allowUnfree = true;
        config.rocmSupport = true;
      };
      lib = pkgs.lib;

      inherit (pkgs.rocmPackages) clr miopen miopengemm rccl rocm-runtime;

      rocm = pkgs.symlinkJoin {
        name = "rocm";
        paths = with pkgs.rocmPackages; [ 
          rocm-thunk 
          rocm-runtime 
          rocm-device-libs 
          clr 
          hipcc
          rccl
          llvm.clang
          miopen
          miopengemm
          miopen-hip
          hipblas
          rocm-cmake
          hip-common
        ];
      };

      mkShell = pkgs.mkShell.override {
        stdenv = pkgs.rocmPackages.llvm.rocmClangStdenv;
      };
    in 
    {
      packages = {
        legion = pkgs.callPackage ./.flake/pkgs/legion.nix { };
        hpp2plantuml = pkgs.python3Packages.callPackage ./.flake/pkgs/hpp2plantuml.nix { };
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
            export HIP_COMPILER="${pkgs.rocmPackages.llvm.clang}/bin/clang"
            export PATH="$HOME/ff/.scripts/:$PATH"
            export ROCM_PATH=${clr}
            export HIP_DEVICE_LIB_PATH="${pkgs.rocmPackages.rocm-device-libs}/amdgcn/bitcode"
            # export HIP_ROOT_DIR=${clr}
            # export HIP_PATH=${clr}/hip
            # export HIP_INCLUDE_DIRS=${clr}/hip/include
            echo "ROCm path set to: $ROCM_PATH"
          '';
          
          CMAKE_FLAGS = lib.strings.concatStringsSep " " [
            "-DFF_USE_EXTERNAL_LEGION=ON"
            "-DFF_USE_EXTERNAL_NCCL=ON"
            "-DFF_USE_EXTERNAL_JSON=ON"
            "-DFF_USE_EXTERNAL_FMT=ON"
            "-DFF_USE_EXTERNAL_SPDLOG=ON"
            "-DFF_USE_EXTERNAL_DOCTEST=ON"
            "-DFF_USE_EXTERNAL_RAPIDCHECK=ON"
            "-DFF_USE_EXTERNAL_EXPECTED=ON"
            "-DFF_USE_EXTERNAL_RANGEV3=ON"
            "-DFF_USE_EXTERNAL_BOOST_PREPROCESSOR=ON"
            "-DFF_USE_EXTERNAL_TYPE_INDEX=ON"

            # hip related flags
            "-DHIP_PLATFORM=amd"
            # "-DHIP_RUNTIME=rocclr"
            # "-DHIP_COMPILER=${pkgs.rocmPackages.llvm.clang}/bin/clang"
            "-DHIP_PATH=${clr}/hip"
            "-DHIP_ROOT_DIR=${clr}/hip"

          ];

          RC_PARAMS = "max_discard_ratio=100";

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
              tl-expected
            ])
            (with self.packages.${system}; [
              legion
              hpp2plantuml
              rapidcheckFull
              doctest
            ])
            (with pkgs.rocmPackages; [
              clr
              miopen
              miopengemm
              rccl
              rocm-runtime
              hipblas
              hipcc
              hip-common
              rocm-cmake
              miopen-hip
              rocm-thunk
              rocm-device-libs
            ])
            # [ rocm ]
          ];
        };

        default = mkShell {
          inputsFrom = [ ci ];
          inherit (ci) CMAKE_FLAGS RC_PARAMS;

          VIMPLUGINS = lib.strings.concatStringsSep "," [
            "${proj-repo.packages.${system}.proj-nvim}"
          ];

          buildInputs = builtins.concatLists [
            (with pkgs; [
              clang-tools
              gh-markdown-preview
              shellcheck
              plantuml
              gdb
              ruff
              compdb
              jq
              gh
            ])
            (with proj-repo.packages.${system}; [
              proj
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
            ])
          ];
        };
      };
    }
  );
}
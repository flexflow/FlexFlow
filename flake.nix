{
  description = "A framework for automatic performance optimization of DNN training and inference";

  nixConfig = {
    bash-prompt-prefix = "(ff) ";
    extra-substituters = [
      "https://ff.cachix.org"
      "https://cuda-maintainers.cachix.org/"
    ];
    extra-trusted-public-keys = [
      "cuda-maintainers.cachix.org-1:0dq3bujKpuEPMCX6U4WylrUDZ9JyUG0VpVZa7CNfq5E="
      "ff.cachix.org-1:/kyZ0w35ToSJBjpiNfPLrL3zTjuPkUiqf2WH0GIShXM="
    ];
  };

  inputs = {
    nixpkgs.url = "nixpkgs/nixos-23.11";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils, ... }: flake-utils.lib.eachSystem [ "x86_64-linux" ] (system: 
    let 
      pkgs = import nixpkgs {
        inherit system;
        config.allowUnfree = true;
      };

      mkShell = pkgs.mkShell.override {
        stdenv = pkgs.cudaPackages.backendStdenv;
      };
    in 
    {
      packages = {
        legion = pkgs.callPackage ./.flake/pkgs/legion.nix { };
        rapidcheckFull = pkgs.symlinkJoin {
          name = "rapidcheckFull";
          paths = (with pkgs; [ rapidcheck.out rapidcheck.dev ]);
        };
      };

      devShells = rec {
        ci = mkShell {
          buildInputs = builtins.concatLists [
            (with pkgs; [
              zlib
              nlohmann_json
              spdlog
              range-v3
              doctest
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
            ])
            (with self.packages.${system}; [
              legion
              rapidcheckFull
            ])
          ];
        };

        default = mkShell {
          inputsFrom = [ ci ];

          buildInputs = builtins.concatLists [
            (with pkgs; [
              ccls
              gh-markdown-preview
              plantuml
              gdb
              ruff
              compdb
              jq
              gh
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

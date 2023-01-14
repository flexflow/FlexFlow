# Resources
# ---
# https://github.com/NixOS/templates/blob/2d6dcce2f3898090c8eda16a16abdff8a80e8ebf/c-hello/flake.nix
#
# According to https://nixos.wiki/wiki/CUDA, it is recommended to enable the cuda-maintainers cachix instance, i.e., 
# add 
# ```
# nix.settings = {
#   substituters = [
#     "https://cuda-maintainers.cachix.org/"
#   ];
#   trusted-public-keys = [
#     "cuda-maintainers.cachix.org-1:0dq3bujKpuEPMCX6U4WylrUDZ9JyUG0VpVZa7CNfq5E="
#   ];
# };
# ```
# to /etc/nixos/configuration.nix.

{
  description = "A framework for automatic performance optimization of DNN training and inference";

  nixConfig.bash-prompt-prefix = "(ff) ";

  # Nixpkgs / NixOS version to use.
  inputs = {
    nixpkgs.url = "nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils, ... }: flake-utils.lib.eachSystem [ "x86_64-linux" ] (system: 
    let 
      pkgs = import nixpkgs {
        inherit system;
        config.allowUnfree = true;
      };
    in 
      {
        devShell = pkgs.mkShell {
          buildInputs = with pkgs; [
            cmake      
            clang
            clangStdenv
            ccache
            cudatoolkit
            cudaPackages.cuda_nvcc
            cudaPackages.cudnn
            cudaPackages.nccl
            cudaPackages.libcublas
            cudaPackages.cuda_cudart
            gdb
            zlib
            pkg-config
            bashInteractive
            python3
          ];
        };
      }
  );
}

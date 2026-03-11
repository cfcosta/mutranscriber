{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
    pre-commit-hooks = {
      url = "github:cachix/pre-commit-hooks.nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
    treefmt-nix = {
      url = "github:numtide/treefmt-nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
    rust-overlay = {
      url = "github:oxalica/rust-overlay";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs =
    {
      nixpkgs,
      flake-utils,
      pre-commit-hooks,
      rust-overlay,
      treefmt-nix,
      ...
    }:
    flake-utils.lib.eachDefaultSystem (
      system:
      let
        pkgs = import nixpkgs {
          inherit system;
          overlays = [ (import rust-overlay) ];
          config.allowUnfree = true;
        };
        inherit (pkgs) mkShell;

        rust = pkgs.rust-bin.fromRustupToolchainFile ./rust-toolchain.toml;

        rustPlatform = pkgs.makeRustPlatform {
          rustc = rust;
          cargo = rust;
        };

        formatter =
          (treefmt-nix.lib.evalModule pkgs {
            projectRootFile = "flake.nix";

            settings = {
              allow-missing-formatter = true;
              verbose = 0;

              global.excludes = [ "*.lock" ];

              formatter = {
                nixfmt.options = [ "--strict" ];
                rustfmt.package = rust;
              };
            };

            programs = {
              nixfmt.enable = true;
              prettier.enable = true;
              rustfmt = {
                enable = true;
                package = rust;
              };
              taplo.enable = true;
            };
          }).config.build.wrapper;

        cuda = pkgs.symlinkJoin {
          name = "cuda-redist";
          paths = with pkgs.cudaPackages; [
            cuda_cudart
            cuda_nvcc
            cudnn
            pkgs.cudatoolkit
          ];
        };

        # Pre-fetched cutlass headers for candle-flash-attn (can't git clone in Nix sandbox)
        cutlass-src = pkgs.fetchFromGitHub {
          owner = "NVIDIA";
          repo = "cutlass";
          rev = "7d49e6c7e2f8896c47f586706e67e1fb215529dc";
          hash = "sha256-D/s7eYsa5l/mfx73tE4mnFcTQdYqGmXa9d9TCryw4e4=";
        };
      in
      {
        checks.pre-commit-check = pre-commit-hooks.lib.${system}.run {
          src = ./.;

          hooks = {
            deadnix.enable = true;
            nixfmt-rfc-style.enable = true;
            treefmt = {
              enable = true;
              package = formatter;
            };
          };
        };

        devShells.default = mkShell {
          name = "mutranscriber";

          buildInputs = [
            cuda
            formatter
            rust

            pkgs.bacon
            pkgs.cargo-machete
            pkgs.cargo-nextest

            pkgs.alsa-lib
            pkgs.gst_all_1.gst-libav
            pkgs.gst_all_1.gst-plugins-bad
            pkgs.gst_all_1.gst-plugins-base
            pkgs.gst_all_1.gst-plugins-good
            pkgs.gst_all_1.gst-plugins-ugly
            pkgs.gst_all_1.gstreamer
            pkgs.libinput
            pkgs.libudev-zero
            pkgs.pipewire
            pkgs.pkg-config
          ];

          CUDA_HOME = if pkgs.stdenv.isLinux then "${cuda}" else "";
          CUDA_PATH = if pkgs.stdenv.isLinux then "${cuda}" else "";

          shellHook = ''
            export LD_LIBRARY_PATH=${cuda}/lib64:${cuda}/lib:$LD_LIBRARY_PATH
          '';
        };

        formatter = formatter;

        packages = rec {
          # CPU-only build (default)
          mutranscriber-cpu = rustPlatform.buildRustPackage {
            pname = "mutranscriber";
            version = "0.1.0";
            src = ./.;
            cargoLock.lockFile = ./Cargo.lock;

            # Skip tests during Nix build (integration tests require model download)
            doCheck = false;

            nativeBuildInputs = [
              pkgs.pkg-config
              pkgs.makeWrapper
            ];
            buildInputs = [
              pkgs.gst_all_1.gstreamer
              pkgs.gst_all_1.gst-plugins-base
              pkgs.gst_all_1.gst-plugins-good
              pkgs.gst_all_1.gst-plugins-bad
              pkgs.gst_all_1.gst-plugins-ugly
              pkgs.gst_all_1.gst-libav
            ];

            # GStreamer plugin path for runtime
            postInstall = ''
              wrapProgram $out/bin/mutranscriber \
                --prefix GST_PLUGIN_PATH : "${pkgs.gst_all_1.gstreamer}/lib/gstreamer-1.0" \
                --prefix GST_PLUGIN_PATH : "${pkgs.gst_all_1.gst-plugins-base}/lib/gstreamer-1.0" \
                --prefix GST_PLUGIN_PATH : "${pkgs.gst_all_1.gst-plugins-good}/lib/gstreamer-1.0" \
                --prefix GST_PLUGIN_PATH : "${pkgs.gst_all_1.gst-plugins-bad}/lib/gstreamer-1.0" \
                --prefix GST_PLUGIN_PATH : "${pkgs.gst_all_1.gst-plugins-ugly}/lib/gstreamer-1.0" \
                --prefix GST_PLUGIN_PATH : "${pkgs.gst_all_1.gst-libav}/lib/gstreamer-1.0"
            '';

            meta = {
              description = "Audio transcription using Qwen3-ASR";
              license = pkgs.lib.licenses.mit;
              mainProgram = "mutranscriber";
            };
          };

          # CUDA-enabled build
          mutranscriber-cuda = rustPlatform.buildRustPackage {
            pname = "mutranscriber-cuda";
            version = "0.1.0";
            src = ./.;
            cargoLock.lockFile = ./Cargo.lock;

            # Skip tests during Nix build
            doCheck = false;

            nativeBuildInputs = [
              pkgs.pkg-config
              pkgs.makeWrapper
              pkgs.cudaPackages.cuda_nvcc
              pkgs.git
            ];

            buildInputs = [
              cuda

              pkgs.autoAddDriverRunpath
              pkgs.gst_all_1.gst-libav
              pkgs.gst_all_1.gst-plugins-bad
              pkgs.gst_all_1.gst-plugins-base
              pkgs.gst_all_1.gst-plugins-good
              pkgs.gst_all_1.gst-plugins-ugly
              pkgs.gst_all_1.gstreamer
            ];

            # Enable CUDA feature
            buildFeatures = [ "cuda" ];

            CUDA_HOME = "${cuda}";
            CUDA_PATH = "${cuda}";
            CUDA_ROOT = "${cuda}";

            # Wrap git so candle-flash-attn's build script uses pre-fetched cutlass
            preBuild =
              let
                gitWrapper = pkgs.writeShellScript "git" ''
                  case "$1" in
                    clone)
                      dest="''${@: -1}"
                      mkdir -p "$dest"
                      cp -r "${cutlass-src}/." "$dest/"
                      ${pkgs.git}/bin/git init -q "$dest"
                      cd "$dest"
                      ${pkgs.git}/bin/git add -A
                      ${pkgs.git}/bin/git -c user.name=nix -c user.email=nix@nix commit -q -m init
                      exit 0
                      ;;
                    sparse-checkout|fetch|checkout)
                      exit 0
                      ;;
                    *)
                      exec ${pkgs.git}/bin/git "$@"
                      ;;
                  esac
                '';
              in
              ''
                mkdir -p .git-wrapper
                ln -sf ${gitWrapper} .git-wrapper/git
                export PATH="$PWD/.git-wrapper:$PATH"
              '';

            # Set CUDA compute capability to skip nvidia-smi detection in sandbox
            # Default to 8.6 (Ampere/RTX 30 series). Override with CUDA_COMPUTE_CAP env var if needed.
            CUDA_COMPUTE_CAP = "86";

            # Wrap binary with proper library paths for CUDA driver
            postInstall = ''
              wrapProgram $out/bin/mutranscriber \
                --prefix LD_LIBRARY_PATH : "/run/opengl-driver/lib" \
                --prefix LD_LIBRARY_PATH : "${cuda}/lib" \
                --prefix LD_LIBRARY_PATH : "${cuda}/lib64" \
                --prefix GST_PLUGIN_PATH : "${pkgs.gst_all_1.gstreamer}/lib/gstreamer-1.0" \
                --prefix GST_PLUGIN_PATH : "${pkgs.gst_all_1.gst-plugins-base}/lib/gstreamer-1.0" \
                --prefix GST_PLUGIN_PATH : "${pkgs.gst_all_1.gst-plugins-good}/lib/gstreamer-1.0" \
                --prefix GST_PLUGIN_PATH : "${pkgs.gst_all_1.gst-plugins-bad}/lib/gstreamer-1.0" \
                --prefix GST_PLUGIN_PATH : "${pkgs.gst_all_1.gst-plugins-ugly}/lib/gstreamer-1.0" \
                --prefix GST_PLUGIN_PATH : "${pkgs.gst_all_1.gst-libav}/lib/gstreamer-1.0"
            '';

            meta = {
              description = "Audio transcription using Qwen3-ASR (CUDA-enabled)";
              license = pkgs.lib.licenses.mit;
              defaultProgram = "mutranscriber";
              mainProgram = "mutranscriber";
            };
          };

          # Default to CPU build
          default = mutranscriber-cpu;
        };
      }
    );
}

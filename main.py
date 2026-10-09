import traceback
import argparse
import sys
import signal
# torchlight
import torchlight
from torchlight.io import str2bool
from torchlight.io import DictAction
from torchlight.io import import_class
import torch
def cleanup_cuda():
    if torch.cuda.is_available():
        print("[INFO] Cleaning up CUDA resources...")
        try:
            torch.cuda.empty_cache()
            print("[INFO] CUDA cache cleared.")
        except Exception as e:
            print(f"[WARN] Failed to clear CUDA cache: {e}")
def safe_import(path, processors, key):
    try:
        processors[key] = import_class(path)
    except Exception as e:
        print(f"[Warning] Failed to import '{key}' from '{path}': {e}")
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Processors')

    # region register processor yapf: disable

    processors = {}

    safe_import('utils.assessment.ASESS_processor', processors, 'assess')
    safe_import('utils.assessment_target.ASESS_processor', processors, 'assess_target')

    subparsers = parser.add_subparsers(dest='processor')

    """
    processors
    k
    'assess'。
    p
    """
    try:
        for k, p in processors.items():
            subparsers.add_parser(k, parents=[p.get_parser()])

        # read arguments
        arg = parser.parse_args()

        # start processor
        Processor = processors[arg.processor]
        p = Processor(sys.argv[2:])

        # handle Ctrl+C
        def handle_sigint(sig, frame):
            print("\n[INFO] Caught Ctrl+C, shutting down gracefully...")
            try:
                p.stop()  # gracefully stop processor if possible
            except Exception as e:
                print(f"[WARN] Failed to stop processor: {e}")
            cleanup_cuda()
            sys.exit(0)

        signal.signal(signal.SIGINT, handle_sigint)

        p.start()

    except KeyboardInterrupt:
        print("\n[INFO] KeyboardInterrupt received, shutting down gracefully...")
        try:
            p.stop()
        except Exception as e:
            print(f"[WARN] Failed to stop processor: {e}")
        cleanup_cuda()
        sys.exit(0)

    except Exception as e:
        print(f"[ERROR] Unhandled exception: {e}")
        traceback.print_exc()
        cleanup_cuda()
        sys.exit(1)

import argparse
from .detect import detect
from .calibrate import calibrate


def main():
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument("--dictionary-id", type=int, default=2)

    subparsers = argument_parser.add_subparsers()

    # detect
    parser_detect = subparsers.add_parser("detect")
    parser_detect.add_argument("src", type=str)
    parser_detect.add_argument("dst", type=str)
    parser_detect.add_argument("camera_params", type=str)
    parser_detect.add_argument("--marker-ids", required=True, nargs="*", type=int)
    parser_detect.add_argument("--marker-length", type=float, default=20e-3)
    parser_detect.add_argument("--detector-params", "-dp", type=str, default=None)
    parser_detect.add_argument("--show-img", action="store_true")
    parser_detect.set_defaults(handler=command_detect)

    # calibrate
    parser_calibrate = subparsers.add_parser("calibrate")
    parser_calibrate.add_argument("src", type=str)
    parser_calibrate.add_argument("dst", type=str)
    parser_calibrate.add_argument("--squares-x", "-sx", type=int, default=8)
    parser_calibrate.add_argument("--squares-y", "-sy", type=int, default=11)
    parser_calibrate.add_argument("--square-length", "-sl", type=float, default=15e-3)
    parser_calibrate.add_argument("--marker-length", "-ml", type=float, default=11e-3)
    parser_calibrate.add_argument("--detector-params", "-dp", type=str, default=None)
    parser_calibrate.add_argument("--show-img", action="store_true")
    parser_calibrate.set_defaults(handler=command_calibrate)

    args = argument_parser.parse_args()
    if hasattr(args, "handler"):
        args.handler(args)
    else:
        argument_parser.print_help()


def command_detect(args: argparse.Namespace):
    detect(
        args.src,
        args.dst,
        args.camera_params,
        args.marker_ids,
        args.dictionary_id,
        args.marker_length,
        args.detector_params,
        args.show_img,
    )


def command_calibrate(args: argparse.Namespace):
    calibrate(
        args.src,
        args.dst,
        args.dictionary_id,
        args.squares_x,
        args.squares_y,
        args.square_length,
        args.marker_length,
        args.detector_params,
        args.show_img,
    )


if __name__ == "__main__":
    main()

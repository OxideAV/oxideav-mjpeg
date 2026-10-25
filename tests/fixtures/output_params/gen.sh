#!/bin/sh
# Regenerates the tests/output_params.rs fixtures with black-box ffmpeg
# (its mjpeg encoder, the codec ffmpeg writes into AVI as `MJPG`) and
# prints what ffprobe reports for each AVI, which is what the test
# pins. The test reads the raw Motion-JPEG twins (`-f mjpeg`, the same
# SOI..EOI frames the AVI carries, one per packet). Run from this
# directory.
set -e
enc() {
    name=$1; shift
    ffmpeg -hide_banner -loglevel error -y "$@" -c:v mjpeg -q:v 8 "$name.avi"
    ffmpeg -hide_banner -loglevel error -y -i "$name.avi" -c:v copy -f mjpeg "$name.mjpeg"
    printf '%s: ' "$name.avi"
    ffprobe -v error -show_streams -of compact "$name.avi" | tr '|' '\n' |
        grep -E '^(width|height|pix_fmt|color_range|color_space)=' | tr '\n' ' '
    echo
    rm -f "$name.avi"
}
enc yuvj420p_70x46 -f lavfi -i testsrc=size=70x46:rate=25 -frames:v 2 -pix_fmt yuvj420p
enc yuvj422p_48x30 -f lavfi -i testsrc=size=48x30:rate=25 -frames:v 2 -pix_fmt yuvj422p
enc yuvj444p_33x17 -f lavfi -i testsrc=size=33x17:rate=25 -frames:v 2 -pix_fmt yuvj444p
# Grayscale, 12-bit, RGB-coded and lossless layouts ffmpeg's encoder
# does not write are encoded in-process by tests/output_params.rs with
# this crate's own encoder; OXIDEAV_MJPEG_OUTPUT_PARAMS_DUMP=<dir>
# writes them out as .jpg for the same ffprobe check.

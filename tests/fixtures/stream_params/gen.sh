#!/bin/sh
# Regenerates the tests/stream_params.rs fixture with black-box ffmpeg:
# three Motion-JPEG segments of different size / sampling (its mjpeg
# encoder, 2 frames each) spliced back to back with the concat demuxer
# into one raw Motion-JPEG stream (`-c copy -f mjpeg`, the SOI..EOI
# frames as an AVI `MJPG` stream carries them), then the per-frame
# ffprobe view of the splice the test pins. Run from this directory.
set -e
tmp=$(mktemp -d)
seg() {
    ffmpeg -hide_banner -loglevel error -y -f lavfi -i "testsrc=size=$2:rate=25" \
        -frames:v 2 -pix_fmt "$3" -c:v mjpeg -q:v 8 "$tmp/$1.avi"
    echo "file '$tmp/$1.avi'" >> "$tmp/list.txt"
}
seg a 40x24 yuvj420p
seg b 64x32 yuvj422p
seg c 40x24 yuvj444p
seg d 40x24 yuvj420p
ffmpeg -hide_banner -loglevel error -y -f concat -safe 0 -i "$tmp/list.txt" \
    -c copy -f mjpeg splice.mjpeg
ffprobe -v error -show_frames -of compact \
    -show_entries frame=width,height,pix_fmt splice.mjpeg
rm -rf "$tmp"

# Demo assets

## `sample_faces.mp4`

- Source: [intel-iot-devkit/sample-videos](https://github.com/intel-iot-devkit/sample-videos) —
  `face-demographics-walking-and-pause.mp4`
- License: [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/)
- Attribution: Intel IoT DevKit sample-videos.
- Changes: the file is redistributed unmodified. The app only plays two trimmed
  frame ranges (418–522 and 660–840) as demo scenes.

## `demo_faces/a.jpg` … `d.jpg`, `demo_face.jpg`

- Derived from frames of the video above by cropping and resizing.
- License: CC BY 4.0, with the same attribution.
- Purpose: seed the local demo gallery (`state/gallery.json`) on first run. The
  gallery itself is runtime data and is not committed.

## Notes

The demo media is **not** covered by this repository's AGPL-3.0 code license.
It is included only to make the teaching demo runnable out of the box. If you
redistribute the demo, keep the CC BY 4.0 attribution above.

The clip contains real people filmed for a public computer-vision sample. Do
not use it for surveillance, identity profiling or any purpose that violates
applicable privacy law. If you are a rights holder and want an asset removed,
please open an issue.

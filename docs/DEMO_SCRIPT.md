# Local demonstration

1. Start Streamlit and open Home. Check the checkpoint and data indicators. Inspect the RDD summary after running the inspection command.
2. Open Environment Lab. Upload one road photo or use a prepared validation image. With no images available, use the calibration pattern. Change darkness or fog intensity and repeat with the same seed. Compare the actual transformed pixels.
3. Open Detect with trained weights. Choose a class and a confidence cutoff. Inspect the annotated image and table and download the results. A valid run may return no detections.
4. Select Video and upload a short MP4 under 24 MB. Play, pause, then resume. The app processes at most 90 frames. Pause or finish to export the processed segment and frame records.
5. In Fusion Lab, compare RGB, proxy and both branches. Inspect view weights and match provenance. Reduce one health slider to zero, then both.
6. Use Virtual Edge Device. Process frames below 70°C, above 70°C, then above 85°C. Allow two controller updates between switches. Reduce temperature again to observe hysteresis. Compare measured latency and branch count in the run history.
7. In CARD Explorer, inspect access instructions or an authorized local frame group. Compare raw stereo previews and undistorted left-view metric depth. Black/missing points have no measurement.
8. Open Experiments / Results. Inspect per-class performance, false positives and latency for the same labeled frames. Export the recorded run. About / Methods / Limits includes the mapping, licenses and limitations.

For the paired weather demonstration, follow the **Run the paired environment benchmark** commands in README, then select the run in Experiments / Results. Compare daylight and noise for the same frame and export the summary. In Fusion Lab, use the connection presets, examine the branch indices and final cutoff decisions, then restore both streams. In Virtual Edge Device, start the default 50–90°C ramp, pause and resume, then process manual cooling steps at 80°C and 65°C. Compare branch counts and resolutions in the timeline and reset before the next demonstration.

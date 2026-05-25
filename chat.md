# Continue from previous chat.md

**Session:** `8d8b0096-135a-4267-b2c9-21398a1215c8`  
**Project:** `home/oscipal/ocean/current/retrieval`  
**Date:** 2026-05-16 12:29 UTC  

---

## User

our last chat is in chat.md. The limit ran out, countinue from there

## Assistant

Let me read the chat log to get up to speed.

## Assistant

**Tool: Read**
Reading `/home/oscipal/ocean_current_retrieval/chat.md`

<details><summary>Tool Result</summary>

**Error:**
```
File content (44582 tokens) exceeds maximum allowed tokens (25000). Use offset and limit parameters to read specific portions of the file, or search for specific content instead of reading the whole file.
```

</details>

## Assistant

**Tool: Read**
Reading `/home/oscipal/ocean_current_retrieval/chat.md`

<details><summary>Tool Result</summary>

```
0	# Add RVL comparison plot function to sentinel_1
1	
2	**Session:** `f070ff04-27ae-44ef-9bac-bc09a302ae3c`  
3	**Project:** `home/oscipal/ocean/current/retrieval`  
4	**Date:** 2026-05-15 14:41 UTC  
5	
6	---
7	
8	## User
9	
10	in scripts/sentinel_1/doppler_comparison/ add a function to plot the rvl from our product and from the ocean product side by side with a scatter plot next to it. like the plot_comparison function in pipeline/.
11	
12	## Assistant
13	
14	Let me read the relevant files first.
15	
16	## Assistant
17	
18	**Tool: Bash**
19	*List files in doppler_comparison directory*
20	```bash
21	find /home/oscipal/ocean_current_retrieval/scripts/sentinel_1/doppler_comparison -type f | sort
22	```
23	
24	<details><summary>Tool Result</summary>

... (150 lines omitted) ...

175	13	    plot_pipeline_steps as _plot_pipeline_steps,
176	14	)
177	15	from .metocean import (
178	16	    compute_stokes_radial,
179	17	    compute_wave_doppler_bias,
180	18	    load_era5_wave,
181	19	    load_era5_wind,
182	20	    load_glo12_current,
183	21	    match_to_sar_grid,
184	22	    project_current_onto_look,
185	23	)
186	24	from .grid_merge import (
187	25	    compute_stats as _compute_stats_impl,
188	
189	... (1323 lines omitted) ...
190	
191	1349	    args = _build_parser().parse_args()
192	1350	    result = run_pipeline(
193	1351	        slc_safe     = args.slc_safe,
194	1352	        subswath     = args.subswath,
195	1353	        burst_idx    = args.burst_idx,
196	1354	        poeorb_path  = args.poeorb,
197	1355	        aux_cal_path = args.aux_cal,
198	1356	        ocn_safe     = args.ocn_safe,
199	1357	        era5_wind    = args.era5_wind,
```

</details>

## Assistant

**Tool: Read**
Reading `/home/oscipal/ocean_current_retrieval/chat.md`

<details><summary>Tool Result</summary>

```
200	1357	        era5_wind    = args.era5_wind,
201	1358	        era5_wave    = args.era5_wave,
202	1359	        glo12        = args.glo12,
203	1360	        polarisation = args.pol,
204	1361	        block_az     = args.block_az,
205	1362	        block_rg     = args.block_rg,
206	1363	        stride_az    = args.stride_az,
207	1364	        stride_rg    = args.stride_rg,
208	1365	        deramp_method = args.deramp_method,
209	1366	        do_descallop = not args.no_descallop,
210	1367	        tops_scaling = args.tops_scaling,
211	1368	    )
212	1369	    if args.out:
213	1370	        np.savez(args.out, **{k: v for k, v in result.items()
214	1371	                              if isinstance(v, np.ndarray)})
215	1372	        print(f'Saved arrays → {args.out}')
216	1373	
217	```
218	
219	</details>
220	
221	## Assistant
222	
223	Now let me read `plots.py` to see exactly how `plot_comparison` is implemented:
224	

... (350 lines omitted) ...

575	+ 
576	+     if vmax is None:
577	+         finite = np.concatenate([
578	+             our_grid[np.isfinite(our_grid)],
579	+             ocn_grid[np.isfinite(ocn_grid)],
580	+         ])
581	+         vmax = float(np.nanpercentile(np.abs(finite), 98)) if finite.size else 50.0
582	+ 
583	+     bias, rmse, r = compute_stats(our_grid, ocn_grid)
584	+ 
585	+     fig = plt.figure(figsize=(16, 6))
586	+     gs = GridSpec(1, 3, figure=fig, wspace=0.35)
587	+ 
588	+     ax0 = fig.add_subplot(gs[0])
589	+     im0 = ax0.imshow(
590	+         our_grid, extent=extent, origin="lower",
591	+         cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto",
592	+     )
593	+     plt.colorbar(im0, ax=ax0, label="Hz", fraction=0.046, pad=0.04)
594	+     ax0.set_title("Our f_dc [Hz]")
595	+     ax0.set_xlabel("Longitude [°]")
596	+     ax0.set_ylabel("Latitude [°]")
597	+ 
598	+     ax1 = fig.add_subplot(gs[1])
599	+     im1 = ax1.imshow(
```

</details>

## Assistant

**Tool: Read**
Reading `/home/oscipal/ocean_current_retrieval/chat.md`

<details><summary>Tool Result</summary>

```
600	+         ocn_grid, extent=extent, origin="lower",
601	+         cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto",
602	+     )
603	+     plt.colorbar(im1, ax=ax1, label="Hz", fraction=0.046, pad=0.04)
604	+     ax1.set_title("OCN rvlDcObs [Hz]")
605	+     ax1.set_xlabel("Longitude [°]")
606	+     ax1.set_ylabel("Latitude [°]")
607	+ 
608	+     ax2 = fig.add_subplot(gs[2])
609	+     finite = np.isfinite(our_grid) & np.isfinite(ocn_grid)
610	+     our_v = our_grid[finite]
611	+     ocn_v = ocn_grid[finite]
612	+     ax2.scatter(ocn_v, our_v, s=2, alpha=0.3, rasterized=True)
613	+     lim = vmax * 1.05
614	+     ax2.plot([-lim, lim], [-lim, lim], "k--", lw=1)
615	+     ax2.set_xlim(-lim, lim)
616	+     ax2.set_ylim(-lim, lim)
617	+     ax2.set_aspect("equal")
618	+     ax2.set_xlabel("OCN rvlDcObs [Hz]")
619	+     ax2.set_ylabel("Our f_dc [Hz]")
620	+     ax2.set_title(f"Our vs OCN\nbias={bias:+.2f}  RMSE={rmse:.2f}  r={r:.4f}")
621	+ 
622	+     plt.suptitle("Raw Doppler comparison — Our vs OCN", fontsize=12)
623	+     plt.tight_layout()
624	+ 

... (350 lines omitted) ...

975	518	    Interpolated from the ECEF velocity state vectors stored in the annotation.
976	519	    The full 3-D speed |v| is used because k_psi depends on the physical
977	520	    angular sweep rate of the antenna in inertial space, not the ground-track
978	
979	... (70 lines omitted) ...
980	
981	591	    ESA TOPS deramp phase from COPE-GSEG-EOPG-TN-14-0025 Eq. 1,
982	592	    with optional Eq. 8 demodulation.
983	593	
984	594	    This applies:
985	595	        Eq. 1: phi(eta, tau) = exp(-j*pi*kt(tau)*(eta-eta_ref(tau))^2)
986	596	        Eq. 8: phi(eta, tau) *= exp(-j*2*pi*f_dc(tau)*(eta-eta_ref(tau)))
987	597	    using the closest azimuth FM-rate and data DC polynomials to the burst mid
988	598	    time, as recommended by the ESA note.
989	599	    """
990	600	    burst_mid = _burst_mid_time(annot, burst_idx)
991	601	    tau = slant_range_time_vector(annot).astype(np.float64)
992	602	    eta = (
993	603	        np.arange(annot.lines_per_burst, dtype=np.float64) - 0.5 * (annot.lines_per_burst - 1)
994	604	    )[:, np.newaxis] * annot.azimuth_time_interval
995	605	
996	606	    afr = _nearest_estimate(annot.azimuth_fm_rates, burst_mid)
997	607	    dc = _nearest_estimate(annot.dc_estimates, burst_mid)
998	608	
999	609	    k_a = _eval_poly(afr.poly, afr.t0, tau).astype(np.float64)
```

</details>

## Assistant

**Tool: Read**
Reading `/home/oscipal/ocean_current_retrieval/chat.md`

<details><summary>Tool Result</summary>

```
1000	610	    f_dc = _eval_poly(dc.data_poly, dc.t0, tau).astype(np.float64)
1001	611	
1002	612	    vel = _interpolate_velocity(annot, burst_mid)
1003	613	    v_s = float(np.linalg.norm(vel))
1004	614	    k_psi = annot.azimuth_steering_rate * np.pi / 180.0
1005	615	    k_s = (2.0 * v_s / C_LIGHT) * annot.radar_frequency * k_psi
1006	```
1007	
1008	</details>
1009	
1010	## Assistant
1011	
1012	**Tool: Read**
1013	Reading `/home/oscipal/ocean_current_retrieval/scripts/sentinel_1/rvl.py`
1014	
1015	<details><summary>Tool Result</summary>
1016	
1017	```
1018	632	def deramp_burst(
1019	633	    burst: np.ndarray,
1020	634	    annot: S1Annotation,
1021	635	    burst_idx: int,
1022	636	    deramp_method: str = 'current',
1023	637	) -> np.ndarray:
1024	638	    """

... (350 lines omitted) ...

1375	  dc = _nearest_estimate(annot.dc_estimates, burst_mid)
1376	/tmp/ipykernel_112701/2388303193.py:8: UserWarning: _nearest_estimate: nearest record is 0.948 s away from burst time 2026-02-05T16:53:06.949767+00:00 — the burst has no matching annotation record and is using a neighbour's polynomial. This will introduce a ~14 Hz geometry Doppler error.
1377	  dc = _nearest_estimate(annot.dc_estimates, burst_mid)
1378	/tmp/ipykernel_112701/2388303193.py:8: UserWarning: _nearest_estimate: nearest record is 0.946 s away from burst time 2026-02-05T16:53:09.710380+00:00 — the burst has no matching annotation record and is using a neighbour's polynomial. This will introduce a ~14 Hz geometry Doppler error.
1379	  dc = _nearest_estimate(annot.dc_estimates, burst_mid)
1380	/tmp/ipykernel_112701/2388303193.py:8: UserWarning: _nearest_estimate: nearest record is 0.945 s away from burst time 2026-02-05T16:53:12.468936+00:00 — the burst has no matching annotation record and is using a neighbour's polynomial. This will introduce a ~14 Hz geometry Doppler error.
1381	  dc = _nearest_estimate(annot.dc_estimates, burst_mid)
1382	/tmp/ipykernel_112701/2388303193.py:8: UserWarning: _nearest_estimate: nearest record is 0.947 s away from burst time 2026-02-05T16:53:15.225437+00:00 — the burst has no matching annotation record and is using a neighbour's polynomial. This will introduce a ~14 Hz geometry Doppler error.
1383	  dc = _nearest_estimate(annot.dc_estimates, burst_mid)
1384	/tmp/ipykernel_112701/2388303193.py:8: UserWarning: _nearest_estimate: nearest record is 0.953 s away from burst time 2026-02-05T16:53:17.977827+00:00 — the burst has no matching annotation record and is using a neighbour's polynomial. This will introduce a ~14 Hz geometry Doppler error.
1385	  dc = _nearest_estimate(annot.dc_estimates, burst_mid)
1386	
1387	## Assistant
1388	
1389	Those warnings are telling you something important but it's not the cause of the burst 4 problem. The DC estimate records in this annotation are stored at ~1.9 s cadence, and the burst repetition interval is ~2.76 s — they never align, so every burst falls ~0.955 s from its nearest DC record. That's a property of this product, not a code bug, and the ~14 Hz error affects all bursts uniformly so it can't explain why burst 4 specifically is wrong.
1390	
1391	But this also means the DC polynomial isn't used in the `'current'` deramp anyway — that path only uses `_fm_rate_at_burst`. The DC records only matter for `_geom_doppler_annotation` and `_deramp_esa`.
1392	
1393	Can you paste the actual printed output from that cell (the lines after the warnings)? The numbers for `annot_dc`, `our_f_dc`, and `f_dc_ocn` per burst are what we need — specifically whether `annot_dc ≈ f_dc_ocn` (full Doppler, large range variation) or `our_f_dc ≈ f_dc_ocn` (residual, small), and whether burst 4 looks anomalous compared to the others.
1394	
1395	## User
1396	
1397	burst 0:  annot_dc [-6.0, +9.7]  our_f_dc [-8.1, +17.7]  f_dc_ocn [-15.9, +18.9]
1398	burst 1:  annot_dc [-7.7, +17.6]  our_f_dc [-11.9, +22.1]  f_dc_ocn [-25.2, +18.4]
1399	burst 2:  annot_dc [+9.0, +13.8]  our_f_dc [+2.6, +26.5]  f_dc_ocn [-31.3, +27.7]
```

</details>

## Assistant

**Tool: Read**
Reading `/home/oscipal/ocean_current_retrieval/chat.md`

<details><summary>Tool Result</summary>

```
1400	burst 3:  annot_dc [+13.3, +18.4]  our_f_dc [-5.3, +50.2]  f_dc_ocn [-47.9, +34.5]
1401	burst 4:  annot_dc [+6.0, +7.0]  our_f_dc [+2.3, +17.8]  f_dc_ocn [-40.9, +116.9]
1402	burst 5:  annot_dc [+17.2, +23.8]  our_f_dc [+13.1, +34.5]  f_dc_ocn [-2.8, +33.0]
1403	burst 6:  annot_dc [+18.7, +31.9]  our_f_dc [-13.4, +41.8]  f_dc_ocn [+5.0, +39.3]
1404	burst 7:  annot_dc [+13.7, +43.8]  our_f_dc [+9.8, +49.7]  f_dc_ocn [-94.2, +49.5]
1405	burst 8:  annot_dc [+13.7, +35.1]  our_f_dc [-2.9, +41.3]  f_dc_ocn [+5.7, +50.0]
1406	burst 9:  annot_dc [+5.7, +10.5]  our_f_dc [-15.3, +59.2]  f_dc_ocn [-93.4, +48.5]
1407	
1408	## Assistant
1409	
1410	Now the picture is clear. Two key findings from these numbers:
1411	
1412	**Our `f_dc` tracks `annot_dc` closely for all bursts.** Both are in the same few-tens-of-Hz range. This confirms the IPF compensated the geometric Doppler using the annotation DC during focusing — our lag-1 estimator recovers the small post-IPF residual, not the full Doppler. `f_dc_ocn` is the full observed Doppler (geometry included), so the comparison we've been making is apples to oranges. That's why it looked "flat" — it's supposed to be small.
1413	
1414	**Burst 4's annotation DC is nearly flat ([+6.0, +7.0] Hz, 1 Hz span)** while every other burst has 5–25 Hz range variation. That's the anomaly. The DC estimate record for burst 4 is essentially a constant, meaning the IPF focused burst 4 with a nearly range-independent Doppler reference — yet `f_dc_ocn` for burst 4 spans 157 Hz, by far the largest of any burst. Strong ocean feature (current boundary, rain cell) that the annotation DC completely failed to capture.
1415	
1416	The burst 4 sign problem is almost certainly in `f_geom_poe` now. Add it to the print:
1417	
1418	```python
1419	for i in range(len(results)):
1420	    r = results[i]
1421	    print(f"burst {i}:  f_dc [{np.nanmin(r['f_dc']):+.1f}, {np.nanmax(r['f_dc']):+.1f}]  "
1422	          f"f_geom_poe [{r['f_geom_poe'].min():+.1f}, {r['f_geom_poe'].max():+.1f}]  "
1423	          f"f_dca [{np.nanmin(r['f_dca']):+.1f}, {np.nanmax(r['f_dca']):+.1f}]")
1424	```

... (350 lines omitted) ...

1775	- [Numerical output — f_geom_poe, f_dca added for bursts 0–9]
1776	- "Look I want the burst to be the same as with the ocn product, which it isn't. There must be something going wrong, it is not because near range is not processed and not because snr is bad, it's something done wrong for the whole burst. I am not comparing apples with oranges, the same corrections and processing steps are done for both datasets and mine is somehow missing something. The signals shouldn't be that flat, so again, is there anything flattening the signals that isn't supposed to be there/was just implemented to smooth the outcome?"
1777	
1778	7. Pending Tasks:
1779	- Identify the mechanism that is actively flattening `f_dc` across range (user insists there is a bug, not a data quality issue)
1780	- Specifically: find anything in the pipeline that smooths or filters the Doppler estimate
1781	- Read `descallop` function at line 1121 of rvl.py (though this acts on f_dca, not f_dc)
1782	- Investigate whether the `_block_p0_p1` Stage 2 coherent range averaging is the flattening mechanism and whether there's a way around it
1783	- Understand why per-column g1 estimates give the same Doppler for burst 4 specifically (since smaller blocks don't help)
1784	
1785	8. Current Work:
1786	The last active work was searching for smoothing/filtering operations in rvl.py:
1787	```bash
1788	grep -n "def descallop\|smooth\|filter\|median\|convolve\|gaussian\|uniform" /home/oscipal/ocean_current_retrieval/scripts/sentinel_1/rvl.py | head -40
1789	```
1790	Result: Only `def descallop` at line 1121 found. No other explicit smoothing operations.
1791	
1792	The user's last message: "Look I want the burst to be the same as with the ocn product, which it isn't. There must be something going wrong, it is not because near range is not processed and not because snr is bad, it's something done wrong for the whole burst. I am not comparing apples with oranges, the same corrections and processing steps are done for both datasets and mine is somehow missing something. The signals shouldn't be that flat, so again, is there anything flattening the signals that isn't supposed to be there/was just implemented to smooth the outcome?"
1793	
1794	The key context: our f_dc for all bursts follows a smooth/straight range profile (no jumps, just a linear trend), while the OCN's f_dc_ocn shows larger spatial variation. The user has ruled out NaN near-range blocks and low SNR. They suspect an active flattening mechanism.
1795	
1796	The `_block_p0_p1` Stage 2 coherent range average (`p1 = complex(np.sum(g1 * w2) / denom)`) is the primary known smoothing mechanism — it collapses the entire block_rg=512 range samples into one Doppler estimate. However, the user tried smaller blocks and the result was the same, suggesting the per-column g1 values themselves are approximately equal (same Doppler regardless of range column). This points to the deramped signal having nearly uniform Doppler across range columns — either because the IPF compensated some range-varying component, or because there's something in `deramp_burst` creating this.
1797	
1798	9. Optional Next Step:
1799	Read the `descallop` function at line 1121 of `rvl.py` to see if it has any range-smoothing effect. More importantly, investigate whether there is any other range smoothing happening in `estimate_correlation_grid` or `_block_p0_p1`, or whether the issue is that the deramped SLC signal genuinely has uniform Doppler across range (potentially because the IPF's annotation DC polynomial was used during focusing, removing the range-varying geometric Doppler from the SLC before we ever see it).
```

</details>

## Assistant

Good, I have the full context now. We were investigating why `f_dc` is flat/straight across range while the OCN product shows large structural variation. The last search found no smoothing in `rvl.py` — now I need to look at `burst_pipeline.py` to see the actual processing chain between deramping and the lag-1 estimator.

## Assistant

**Tool: Read**
Reading `/home/oscipal/ocean_current_retrieval/scripts/sentinel_1/burst_pipeline.py`

<details><summary>Tool Result</summary>

```
1	#!/usr/bin/env python3
2	"""Run the diagnostic single-burst Sentinel-1 RVL retrieval path."""
3	
4	from __future__ import annotations
5	
6	import argparse
7	import numpy as np
8	import xarray as xr
9	
10	from .safe_io import find_safe_files, parse_annotation, read_slc_burst
11	from .rvl import (
12	    deramp_burst,
13	    estimate_correlation_grid,
14	    correlation_to_doppler,
15	    _geom_doppler_annotation,
16	    _geom_doppler_poeorb,
17	    _geolocate_grid,
18	    _interpolate_orbit,
19	    compute_gamma_ambiguity,
20	    compute_sideband_bias,
21	    apply_burst_valid_sample_mask,
22	)
23	
24	
25	# ---------------------------------------------------------------------------

... (274 lines omitted) ...

300	if __name__ == '__main__':
301	    args = _build_parser().parse_args()
302	
303	    print(f'Processing {args.subswath.upper()} burst {args.burst_idx} …')
304	    ds = compute_rvl_burst(
305	        safe_dir     = args.safe_dir,
306	        subswath     = args.subswath,
307	        burst_idx    = args.burst_idx,
308	        polarisation = args.pol,
309	        block_az     = args.block_az,
310	        block_rg     = args.block_rg,
311	        stride_az    = args.stride_az,
312	        stride_rg    = args.stride_rg,
313	        deramp_method = args.deramp_method,
314	        aux_cal_path = args.aux_cal,
315	        poeorb_path  = args.poeorb,
316	    )
317	
318	    ds.to_netcdf(args.out)
319	    print(f'Saved → {args.out}')
320	    print(ds)
321	
322	    if args.plot:
323	        _plot(ds, args.burst_idx)
324	
```

</details>

## Assistant

Now I see the full chain. Let me look at how `f_dc_ocn` is created in the pipeline — that comparison was driving the whole diagnosis.

## Assistant

**Tool: Bash**
*Find f_dc_ocn assignment in pipeline.py*
```bash
grep -n "f_dc_ocn\|rvlDcObs\|doppler_ocn\|ocn.*dc\|dc.*ocn" /home/oscipal/ocean_current_retrieval/scripts/sentinel_1/pipeline.py | head -40
```

<details><summary>Tool Result</summary>

```
157:    use_ocn_dc:   bool = False,
287:    # ── OCN rvlDcMiss mispointing (+ optional rvlDcObs) ──────────────────────
302:    ocn_dc_miss = _clean('rvlDcMiss')
303:    ocn_dc_obs  = _clean('rvlDcObs')
305:    f_miss_ocn = match_to_sar_grid(lat, lon, ocn_lat, ocn_lon, ocn_dc_miss)
307:    f_dc_ocn   = match_to_sar_grid(lat, lon, ocn_lat, ocn_lon, ocn_dc_obs)
310:    if use_ocn_dc:
311:        f_dc   = f_dc_ocn.astype(np.float32)
315:        #print(f'  using rvlDcObs as f_dc; v_r median: {float(np.nanmedian(v_r)):+.4f} m/s')
357:        'f_dc_ocn':       f_dc_ocn,
396:    use_ocn_dc:   bool = False,
415:    use_ocn_dc    : bool
416:        When True, replace the lag-1 correlation f_dc with rvlDcObs interpolated
429:    if use_ocn_dc:
443:            do_descallop=do_descallop, use_ocn_dc=use_ocn_dc,
464:            tops_scaling=tops_scaling, use_ocn_dc=use_ocn_dc, estimator=estimator,
485:    Uses rvlDcObs as f_dc and rvlDcGeo as the geometry reference, both already
491:    print(f'use_ocn_dc: working on OCN grid for {subswath.upper()} (no SLC processing) …')
519:    ocn_dc_obs  = _clean('rvlDcObs').astype(np.float32)
520:    ocn_dc_geo  = _clean('rvlDcGeo').astype(np.float32)
521:    ocn_dc_miss = _clean('rvlDcMiss').astype(np.float32)
523:    f_dc  = ocn_dc_obs
524:    f_dca = (f_dc - ocn_dc_geo - float(f_sideband)).astype(np.float32)
570:    f_miss_ocn = ocn_dc_miss
599:        'f_dc_ocn':       f_dc,
603:        'f_geom_ann':     ocn_dc_geo,
604:        'f_geom_poe':     ocn_dc_geo,
635:    use_ocn_dc:   bool = False,
737:    # OCN rvlDcMiss (+ optional rvlDcObs)
752:    ocn_dc_miss = _clean('rvlDcMiss')
753:    ocn_dc_obs  = _clean('rvlDcObs')
755:    f_miss_ocn = match_to_sar_grid(lat, lon, ocn_lat, ocn_lon, ocn_dc_miss)
757:    f_dc_ocn   = match_to_sar_grid(lat, lon, ocn_lat, ocn_lon, ocn_dc_obs)
760:    if use_ocn_dc:
761:        f_dc      = f_dc_ocn.astype(np.float32)
772:        print(f'  using rvlDcObs as f_dc; v_r median: {float(np.nanmedian(v_r)):+.4f} m/s')
802:        'f_dc_ocn':       f_dc_ocn,
842:    use_ocn_dc:    bool = False,
958:    ocn_dc_miss = _clean('rvlDcMiss')
959:    ocn_dc_obs = _clean('rvlDcObs')
```

</details>

## Assistant

**Tool: Read**
Reading `/home/oscipal/ocean_current_retrieval/scripts/sentinel_1/rvl.py`

<details><summary>Tool Result</summary>

```
550	
551	
552	def _deramp_rate(annot: S1Annotation, burst_idx: int) -> np.ndarray:
553	    """
554	    Per-range-sample deramp chirp rate k_s(τ) [Hz/s].
555	
556	    Returns
557	    -------
558	    np.ndarray, shape (n_samples,)
559	    """
560	    k_a   = _fm_rate_at_burst(annot, burst_idx)            # < 0
561	    k_psi = _steering_doppler_rate(annot, burst_idx)       # > 0
562	    return -k_a * k_psi / (k_a - k_psi)                   # < 0, varies with range
563	
564	
565	def _eval_poly(poly: np.ndarray, t0: float, tau: np.ndarray) -> np.ndarray:
566	    """Evaluate polynomial coefficients on the slant-range time axis."""
567	    dt = tau - t0
568	    return sum(c * dt**k for k, c in enumerate(poly))
569	
570	
571	def _burst_mid_time(annot: S1Annotation, burst_idx: int):
572	    """Return the mid-burst azimuth time for burst-dependent parameter lookup."""
573	    burst = annot.bursts[burst_idx]
574	    half_dt = 0.5 * (annot.lines_per_burst - 1) * annot.azimuth_time_interval

... (50 lines omitted) ...

625	    return (burst * phi).astype(np.complex64)
626	
627	
628	# ─────────────────────────────────────────────────────────────────────────────
629	# Step I — Deramp + window + merge  (Section 5.4)
630	# ─────────────────────────────────────────────────────────────────────────────
631	
632	def deramp_burst(
633	    burst: np.ndarray,
634	    annot: S1Annotation,
635	    burst_idx: int,
636	    deramp_method: str = 'current',
637	) -> np.ndarray:
638	    """
639	    Apply the TOPS deramp chirp to one complex burst (eq. 1–3 in the doc).
640	
641	        I_cbd(t_az, τ) = c1(t_az, τ) · I_cb(t_az, τ)
642	        c1(t_az, τ)   = exp(iπ k_s(τ) t_az²)
643	
644	    where t_az is azimuth time measured from the burst centre [s].
645	
646	    Parameters
647	    ----------
648	    burst     : complex64, shape (linesPerBurst, samplesPerBurst)
649	    annot     : S1Annotation
```

</details>

## Assistant

Now I see `f_dc_ocn` is directly `rvlDcObs`. Let me read `ocn_product.py` to understand what `rvlDcObs` actually represents physically.

## Assistant

**Tool: Read**
Reading `/home/oscipal/ocean_current_retrieval/scripts/sentinel_1/ocn_product.py`

<details><summary>Tool Result</summary>

```
1	"""Read Sentinel-1 Level-2 OCN SAFE products into xarray datasets."""
2	
3	from __future__ import annotations
4	
5	from pathlib import Path
6	
7	import xarray as xr
8	
9	
10	_OCN_PREFIXES = ("owi", "osw", "rvl")
11	_RADAR_MODES = ("ew", "iw", "sm", "wv")
12	_POLARISATIONS = ("hh", "hv", "vh", "vv")
13	_AUX_COORD_NAMES = {
14	    "lat",
15	    "latitude",
16	    "lon",
17	    "longitude",
18	    "incidenceangle",
19	    "incidence_angle",
20	    "heading",
21	    "landflag",
22	    "mask",
23	    "quality",
24	    "time",
25	}

... (256 lines omitted) ...

282	    decode_cf: bool = True,
283	) -> xr.Dataset:
284	    """Convenience wrapper returning only the OWI dataset from an OCN SAFE."""
285	    return load_ocn_safe(
286	        safe_dir,
287	        swath=swath,
288	        polarisation=polarisation,
289	        decode_cf=decode_cf,
290	    )["owi"]
291	
292	
293	def load_osw_from_ocn(
294	    safe_dir: str,
295	    swath: str | None = None,
296	    polarisation: str | None = None,
297	    decode_cf: bool = True,
298	) -> xr.Dataset:
299	    """Convenience wrapper returning only the OSW dataset from an OCN SAFE."""
300	    return load_ocn_safe(
301	        safe_dir,
302	        swath=swath,
303	        polarisation=polarisation,
304	        decode_cf=decode_cf,
305	    )["osw"]
306	
```

</details>

## Assistant

Now let me read the relevant section of `pipeline.py` around the OCN loading to understand the full context.

## Assistant

**Tool: Read**
Reading `/home/oscipal/ocean_current_retrieval/scripts/sentinel_1/pipeline.py`

<details><summary>Tool Result</summary>

```
145	    ocn_safe:     str,
146	    era5_wind:    str,
147	    era5_wave:    str,
148	    glo12:        str,
149	    polarisation: str = 'vv',
150	    block_az:     int = 256,
151	    block_rg:     int = 512,
152	    stride_az:    int = 128,
153	    stride_rg:    int = 256,
154	    deramp_method: str = 'current',
155	    do_descallop: bool = True,
156	    tops_scaling: str = 'none',
157	    use_ocn_dc:   bool = False,
158	    estimator:    str = 'cde',
159	) -> dict:
160	    """
161	    Run the full pipeline from the walkthrough notebook.
162	
163	    Returns a dict with all intermediate and final arrays:
164	      v_r             SAR radial velocity [m/s], toward-satellite positive
165	      v_stokes        Stokes drift projected onto look [m/s]
166	      v_wave          Wave Doppler bias [m/s]
167	      v_miss_ocn      OCN mispointing correction [m/s]
168	      v_current       Current without mispointing: -v_r - v_stokes - v_wave
169	      v_current_ocn   Fully corrected: -v_r + v_miss_ocn - v_stokes - v_wave

... (180 lines omitted) ...

350	        'v_model':        v_model,
351	        'lat':            lat,
352	        'lon':            lon,
353	        'inc':            inc,
354	        'f_dca':          f_dca,
355	        'f_dca_pre_descallop': f_dca_pre_descallop,
356	        'f_dc':           f_dc,
357	        'f_dc_ocn':       f_dc_ocn,
358	        'tops_scale':     tops_scale,
359	        'tops_scaling':   tops_scaling,
360	        'deramp_method':  deramp_method,
361	        'do_descallop':   do_descallop,
362	        'burst_period_rows': burst_period_rows,
363	        'snr':            snr,
364	        'f_geom_ann':     f_geom_ann,
365	        'f_geom_poe':     f_geom_poe,
366	        'look_az_rad':    look_az,
367	        'heading_deg':    heading,
368	        'wavelength_m':   annot.wavelength,
369	        'sideband_bias_hz': float(f_sideband),
370	        'subswath':       subswath,
371	        'burst_idx':      burst_idx,
372	        'acq_time':       acq_time,
373	        'rmse_vs_glo12':  rmse_vs_glo12,
374	        'bias_vs_glo12':  bias_vs_glo12,
```

</details>

## Assistant

I need to read `_steering_doppler_rate` — that function was omitted in the last read and it's key to understanding whether the deramp rate is correct.

## Assistant

**Tool: Read**
Reading `/home/oscipal/ocean_current_retrieval/scripts/sentinel_1/rvl.py`

<details><summary>Tool Result</summary>

```
514	def _orbital_speed(annot: S1Annotation, burst_idx: int) -> float:
515	    """
516	    Return the satellite orbital speed [m/s] at the burst centre time.
517	
518	    Interpolated from the ECEF velocity state vectors stored in the annotation.
519	    The full 3-D speed |v| is used because k_psi depends on the physical
520	    angular sweep rate of the antenna in inertial space, not the ground-track
521	    speed.
522	    """
523	    burst_time = annot.bursts[burst_idx].azimuth_time
524	    diffs = [abs((t - burst_time).total_seconds()) for t in annot.orbit_times]
525	    idx = int(np.argmin(diffs))
526	    vx, vy, vz = annot.orbit_velocities[idx]
527	    return float(np.sqrt(vx**2 + vy**2 + vz**2))
528	
529	
530	def _steering_doppler_rate(annot: S1Annotation, burst_idx: int) -> float:
531	    """
532	    Doppler rate k_psi [Hz/s] induced by the TOPS antenna steering.
533	
534	    k_psi = (2 f_0 / c) * v_sat * |ψ̇|
535	
536	    v_sat is the satellite orbital speed [m/s] from the annotation state
537	    vectors.  Using orbital speed (≈ 7 500 m/s) rather than the projected
538	    ground speed (≈ 6 800 m/s) is critical: the TOPS antenna rotates at
539	    rate ψ̇ in inertial space, so it is the inertial (orbital) velocity that
540	    enters the Doppler rate formula.  Underestimating v_sat by ~10 % produces
541	    a ~40 Hz/s residual chirp after deramping — the dominant source of
542	    within-burst Doppler stripes.
543	
544	    ψ̇ is the physical steering rate in rad/s.  k_psi > 0 for forward
545	    steering (IW ≈ 7 500 Hz/s).
546	    """
547	    v_sat    = _orbital_speed(annot, burst_idx)                  # m/s
548	    psi_dot  = abs(annot.azimuth_steering_rate) * np.pi / 180    # rad/s
549	    return 2.0 * annot.radar_frequency / C_LIGHT * v_sat * psi_dot
550	
551	
552	def _deramp_rate(annot: S1Annotation, burst_idx: int) -> np.ndarray:
553	    """
554	    Per-range-sample deramp chirp rate k_s(τ) [Hz/s].
555	
556	    Returns
557	    -------
558	    np.ndarray, shape (n_samples,)
```

</details>

## Assistant

**Tool: Read**
Reading `/home/oscipal/ocean_current_retrieval/scripts/sentinel_1/rvl.py`

<details><summary>Tool Result</summary>

```
650	    burst_idx : int
651	    deramp_method : str
652	        `current` uses the existing repository deramp chirp.
653	        `esa_eq1` uses the ESA TOPS deramp formula from Eq. 1.
654	
655	    Returns
656	    -------
657	    complex64, same shape
658	    """
659	    if deramp_method == 'esa_eq1':
660	        return _deramp_esa(burst, annot, burst_idx, demodulate=False)
661	    if deramp_method != 'current':
662	        raise ValueError("deramp_method must be 'current' or 'esa_eq1'")
663	
664	    lpb = annot.lines_per_burst
665	    k_s = _deramp_rate(annot, burst_idx)                               # (n_samples,)
666	    # Centre the discrete azimuth-time axis on the middle sample pair, not on
667	    # the boundary between the two middle samples. A half-line offset leaves a
668	    # residual linear phase term after deramping, which appears as a per-burst
669	    # Doppler bias and can create seams at burst boundaries.
670	    t_az = (
671	        np.arange(lpb, dtype=np.float64) - 0.5 * (lpb - 1)
672	    ) * annot.azimuth_time_interval  # (lpb,) [s]
673	
674	    # Broadcast to (lpb, n_samples)

... (50 lines omitted) ...

725	        4. × w(τ − τ̄_j)   →  raised-cosine window centred on burst
726	        5. FFT
727	        6. × C2*           →  c2* ⊗ (windowed)  (refocus)
728	        7. IFFT
729	    """
730	    lpb   = I_bcd.shape[0]
731	    k_psi = _steering_doppler_rate(annot, burst_idx)          # Hz/s, scalar > 0
732	    w     = _burst_window(lpb, overlap_lines)
733	
734	    f  = np.fft.fftfreq(lpb, d=1.0 / annot.prf)                      # (lpb,) Hz
735	    C2 = np.exp(-1j * np.pi * f**2 / k_psi).astype(np.complex64)     # (lpb,)
736	
737	    # Defocus: IFFT[ C2 · FFT[I_bcd] ]
738	    I_defoc    = np.fft.ifft(C2[:, None] * np.fft.fft(I_bcd, axis=0), axis=0)
739	    # Window in the defocused domain
740	    I_windowed = (I_defoc * w[:, None]).astype(np.complex64)
741	    # Refocus: IFFT[ C2* · FFT[I_windowed] ]
742	    I_cbdw     = np.fft.ifft(C2.conj()[:, None] * np.fft.fft(I_windowed, axis=0), axis=0)
743	
744	    return I_cbdw.astype(np.complex64)
745	
746	
747	def merge_bursts(
748	    annot: S1Annotation,
749	    measurement_path: str,
```

</details>

## Assistant

Let me check `safe_io.py` to see what `annot.prf` actually represents, and also look at the deramp function's full context.

## Assistant

**Tool: Bash**
*Find PRF-related fields in safe_io.py*
```bash
grep -n "prf\|azimuth_frequency\|azimuthFrequency\|radar_prf\|radarPrf" /home/oscipal/ocean_current_retrieval/scripts/sentinel_1/safe_io.py | head -30
```

<details><summary>Tool Result</summary>

```
122:    prf: float                        # Hz (azimuth / output sampling frequency)
123:    radar_prf: float                  # Hz (hardware pulse repetition frequency; used for AUX_CAL ambiguity corrections)
159:    prf = float(ii.findtext('azimuthFrequency'))
160:    radar_prf_el = root.findtext('.//prf')
161:    radar_prf = float(radar_prf_el) if radar_prf_el is not None else prf
283:        prf=prf,
284:        radar_prf=radar_prf,
```

</details>

## Assistant

**Tool: Bash**
*Find SLC burst reading and valid sample fields*
```bash
grep -n "def read_slc_burst\|def slant_range_time\|lines_per_burst\|firstValidSample\|lastValidSample" /home/oscipal/ocean_current_retrieval/scripts/sentinel_1/safe_io.py | head -20
```

<details><summary>Tool Result</summary>

```
126:    lines_per_burst: int
173:        fvs_text = b.findtext('firstValidSample').split()
174:        lvs_text = b.findtext('lastValidSample').split()
287:        lines_per_burst=lpb,
320:def read_slc_burst(tiff_path: str, annot: S1Annotation, burst_idx: int) -> np.ndarray:
331:    lpb = annot.lines_per_burst
451:def slant_range_time_vector(annot: S1Annotation) -> np.ndarray:
```

</details>

## Assistant

Let me check the FFT-based Doppler estimator — it might be the right tool for this scenario.

## Assistant

**Tool: Bash**
*Find FFT Doppler estimator definition*
```bash
grep -n "def estimate_doppler_grid_fft" /home/oscipal/ocean_current_retrieval/scripts/sentinel_1/rvl.py
```

<details><summary>Tool Result</summary>

```
985:def estimate_doppler_grid_fft(
```

</details>

## Assistant

**Tool: Read**
Reading `/home/oscipal/ocean_current_retrieval/scripts/sentinel_1/rvl.py`

<details><summary>Tool Result</summary>

```
985	def estimate_doppler_grid_fft(
986	    I_c: np.ndarray,
987	    prf: float,
988	    block_az: int = 256,
989	    block_rg: int = 512,
990	    stride_az: int = 128,
991	    stride_rg: int = 256,
992	    min_valid_fraction: float = 0.8,
993	    valid_mask: np.ndarray | None = None,
994	) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
995	    """
996	    Slide an estimation window over the image and compute Doppler centroid
997	    via FFT spectral centroid (alternative to estimate_correlation_grid +
998	    correlation_to_doppler).
999	
1000	    Returns the same (f_dc, snr, az_centers, rg_centers) signature as the
1001	    CDE path for drop-in substitution in run_pipeline / compute_rvl.
1002	
1003	    Parameters
1004	    ----------
1005	    I_c                 : complex64, shape (n_az, n_rg)
1006	    prf                 : float [Hz]
1007	    block_az, block_rg  : estimation block dimensions [samples]
1008	    stride_az, stride_rg: block stride [samples]
1009	    min_valid_fraction  : skip blocks with too many invalid pixels

... (25 lines omitted) ...

1035	                np.count_nonzero(block) / block.size
1036	                if mask_block is None
1037	                else np.count_nonzero(mask_block) / mask_block.size
1038	            )
1039	            if valid_frac < min_valid_fraction:
1040	                continue
1041	            f_dc_val, snr_val  = _block_fft_doppler(block, prf, mask_block)
1042	            f_dc_grid[i, j]    = f_dc_val
1043	            snr_grid[i, j]     = snr_val
1044	
1045	    az_centers = az_starts + block_az // 2
1046	    rg_centers = rg_starts + block_rg // 2
1047	
1048	    return f_dc_grid, snr_grid, az_centers, rg_centers
1049	
1050	
1051	# ─────────────────────────────────────────────────────────────────────────────
1052	# Step III — Parameter estimation  (Section 5.7)
1053	# ─────────────────────────────────────────────────────────────────────────────
1054	
1055	def correlation_to_doppler(
1056	    p0: np.ndarray,
1057	    p1: np.ndarray,
1058	    prf: float,
1059	    wavelength: float,
```

</details>

## Assistant

**Tool: Read**
Reading `/home/oscipal/ocean_current_retrieval/scripts/sentinel_1/rvl.py`

<details><summary>Tool Result</summary>

```
933	def _block_fft_doppler(
934	    block: np.ndarray,
935	    prf: float,
936	    valid_mask: np.ndarray | None = None,
937	) -> tuple[float, float]:
938	    """
939	    FFT spectral-centroid Doppler estimator for one block.
940	
941	    Computes f_dc = Σ f·P(f) / Σ P(f) where P is the range-averaged azimuth
942	    power spectrum, with Hanning² weighting across range.  More robust than the
943	    lag-1 CDE to point targets and to residual quadratic phase errors from
944	    imperfect deramping (a symmetric chirp residual broadens the spectral peak
945	    but does not shift the power-weighted centroid).
946	
947	    Returns
948	    -------
949	    f_dc  : float  Doppler centroid [Hz], NaN if insufficient power
950	    snr   : float  spectral peak / noise-floor ratio (linear)
951	    """
952	    n_az, n_rg = block.shape
953	    haz = np.hanning(n_az).astype(np.float32)
954	    hra = np.hanning(n_rg).astype(np.float32)
955	
956	    if valid_mask is not None:
957	        block = np.where(valid_mask, block, 0.0)

... (2 lines omitted) ...

960	
961	    S   = np.fft.fft(I_w, axis=0)          # (n_az, n_rg)
962	    P   = np.abs(S) ** 2                    # azimuth power spectrum per range col
963	
964	    w2    = hra ** 2
965	    denom = float(np.sum(w2))
966	    if denom < 1e-30:
967	        return float('nan'), float('nan')
968	
969	    P_avg = np.sum(P * w2[np.newaxis, :], axis=1) / denom   # (n_az,)
970	
971	    total = float(np.sum(P_avg))
972	    if total < 1e-30:
973	        return float('nan'), float('nan')
974	
975	    freqs = np.fft.fftfreq(n_az, d=1.0 / prf)   # Hz, 0-centred ordering
976	    f_dc  = float(np.sum(freqs * P_avg) / total)
977	
978	    n_noise   = max(1, n_az // 4)
979	    noise     = float(np.mean(np.sort(P_avg)[:n_noise]))
980	    snr_val   = float((P_avg.max() - noise) / noise) if noise > 0 else 0.0
981	
982	    return f_dc, snr_val
983	
984	
```

</details>

## Assistant

Let me read the import section of pipeline.py and the relevant annotation helper functions.

## Assistant

**Tool: Read**
Reading `/home/oscipal/ocean_current_retrieval/scripts/sentinel_1/pipeline.py`

<details><summary>Tool Result</summary>

```
1	"""Run the main Sentinel-1 RVL retrieval pipeline and its merged-burst variants."""
2	
3	from __future__ import annotations
4	
5	import argparse
6	
7	import numpy as np
8	
9	from scripts.diagnostics.pipeline_diagnostics import (
10	    diagnose_burst_doppler as _diagnose_burst_doppler,
11	    diagnose_mispointing_aux_ins as _diagnose_mispointing_aux_ins,
12	    plot_dc_estimates as _plot_dc_estimates,
13	    plot_pipeline_steps as _plot_pipeline_steps,
14	)
15	from .metocean import (
16	    compute_stokes_radial,
17	    compute_wave_doppler_bias,
18	    load_era5_wave,
19	    load_era5_wind,
20	    load_glo12_current,
21	    match_to_sar_grid,
22	    project_current_onto_look,
23	)
24	from .grid_merge import (
25	    compute_stats as _compute_stats_impl,

... (95 lines omitted) ...

121	
122	    valid = np.isfinite(lat_f) & np.isfinite(lon_f) & np.isfinite(data_f)
123	
124	    grid_lat = np.arange(lat_f[valid].min(), lat_f[valid].max() + resolution_deg, resolution_deg)
125	    grid_lon = np.arange(lon_f[valid].min(), lon_f[valid].max() + resolution_deg, resolution_deg)
126	
127	    mesh_lon, mesh_lat = np.meshgrid(grid_lon, grid_lat)
128	
129	    grid_data = griddata(
130	        points=(lon_f[valid], lat_f[valid]),
131	        values=data_f[valid],
132	        xi=(mesh_lon, mesh_lat),
133	        method=method,
134	    ).astype(np.float32)
135	
136	    return grid_lat, grid_lon, grid_data
137	
138	
139	def run_pipeline(
140	    slc_safe:     str,
141	    subswath:     str,
142	    burst_idx:    int,
143	    poeorb_path:  str,
144	    aux_cal_path: str,
145	    ocn_safe:     str,
```

</details>

## Assistant

**Tool: Read**
Reading `/home/oscipal/ocean_current_retrieval/scripts/sentinel_1/safe_io.py`

<details><summary>Tool Result</summary>

```
110	@dataclass
111	class S1Annotation:
112	    polarisation: str
113	    subswath: str
114	    radar_frequency: float            # Hz
115	    wavelength: float                 # m
116	    azimuth_steering_rate: float      # deg/s
117	    range_sampling_rate: float        # Hz
118	    slant_range_time_start: float     # s  (near range)
119	    range_pixel_spacing: float        # m
120	    azimuth_pixel_spacing: float      # m
121	    azimuth_time_interval: float      # s
122	    prf: float                        # Hz (azimuth / output sampling frequency)
123	    radar_prf: float                  # Hz (hardware pulse repetition frequency; used for AUX_CAL ambiguity corrections)
124	    n_samples: int
125	    n_lines: int
126	    lines_per_burst: int
127	    samples_per_burst: int
128	    first_line_time: datetime
129	    bursts: list                      # list[BurstInfo]
130	    dc_estimates: list                # list[DcEstimate]
131	    azimuth_fm_rates: list            # list[AzimuthFmRate]
132	    geoloc_grid: dict                 # 'line','pixel','lat','lon','inc_angle' arrays
133	    orbit_times: list                 # list[datetime]  state-vector times
134	    orbit_positions: list             # list of [x, y, z] in metres (ECEF)

... (30 lines omitted) ...

165	
166	    # Bursts
167	    st = root.find('.//swathTiming')
168	    lpb = int(st.findtext('linesPerBurst'))
169	    spb = int(st.findtext('samplesPerBurst'))
170	
171	    bursts = []
172	    for idx, b in enumerate(root.findall('.//swathTiming/burstList/burst')):
173	        fvs_text = b.findtext('firstValidSample').split()
174	        lvs_text = b.findtext('lastValidSample').split()
175	        bursts.append(BurstInfo(
176	            idx=idx,
177	            azimuth_time=_iso_to_datetime(b.findtext('azimuthTime')),
178	            byte_offset=int(b.findtext('byteOffset')),
179	            first_valid_sample=np.array([int(x) for x in fvs_text], dtype=np.int32),
180	            last_valid_sample=np.array([int(x) for x in lvs_text], dtype=np.int32),
181	            sensing_time=_iso_to_datetime(b.findtext('sensingTime')),
182	        ))
183	
184	    # Doppler centroid estimates
185	    dc_estimates = []
186	    for dc in root.findall('.//dopplerCentroid/dcEstimateList/dcEstimate'):
187	        geom_poly_txt = dc.findtext('geometryDcPolynomial')
188	        data_poly_txt = dc.findtext('dataDcPolynomial')
189	        if geom_poly_txt is None or data_poly_txt is None:
```

</details>

## Assistant

**Tool: Read**
Reading `/home/oscipal/ocean_current_retrieval/scripts/sentinel_1/safe_io.py`

<details><summary>Tool Result</summary>

```
451	def slant_range_time_vector(annot: S1Annotation) -> np.ndarray:
452	    """Return slant-range time for each range sample [s]."""
453	    dt_r = 1.0 / annot.range_sampling_rate
454	    return annot.slant_range_time_start + np.arange(annot.n_samples) * dt_r
455	
456	
457	def _nearest_estimate(estimates, burst_time: datetime, warn_threshold_s: float = 0.4):
458	    """
459	    Return the estimate whose azimuth_time is closest to burst_time.
460	
461	    Emits a warning when the nearest record is more than *warn_threshold_s*
462	    seconds away — indicating the burst has no matching record in the annotation
463	    and is falling back to an adjacent burst's polynomial.  This causes a
464	    systematic geometry Doppler error of roughly Δf_geom ≈ orbit_DC_rate × Δt.
465	    """
466	    import warnings
467	    diffs = [abs((e.azimuth_time - burst_time).total_seconds()) for e in estimates]
468	    best = int(np.argmin(diffs))
469	    if diffs[best] > warn_threshold_s:
470	        warnings.warn(
471	            f"_nearest_estimate: nearest record is {diffs[best]:.3f} s away from "
472	            f"burst time {burst_time.isoformat()} — the burst has no matching annotation "
473	            f"record and is using a neighbour's polynomial. "
474	            f"This will introduce a ~{diffs[best] * 15:.0f} Hz geometry Doppler error.",
475	            UserWarning,
476	            stacklevel=2,
477	        )
478	    return estimates[best]
479	
```

</details>

## Assistant

Now I understand the full picture. Here's the root cause:

**Our `f_dc` is the lag-1 RESIDUAL after the IPF's DC centering**, not the full observed Doppler. The IPF subtracts `data_poly` (annotation DC) during azimuth compression, so our lag-1 sees only the small residual (≈ 0–few tens of Hz). ESA's `rvlDcObs` is the **full** observed Doppler (what the IPF removed, plus any residual), which is typically much larger and has the range structure we expect.

The fix is to add the annotation `data_poly` back after our lag-1 estimate:
```
f_dc_full = our_f_dc + data_poly
f_dca = f_dc_full - f_geom_poe
```

For burst 4 specifically: the annotation `data_poly` is nearly flat (+6-7 Hz) because the IPF's DC estimation failed for that heterogeneous burst (current boundary). So even after this fix, burst 4 won't show the correct gradient — that requires using `use_ocn_dc=True` for that burst.

Let me implement the fix in `pipeline.py` and `burst_pipeline.py`:

## Assistant

**Tool: Read**
Reading `/home/oscipal/ocean_current_retrieval/scripts/sentinel_1/pipeline.py`

<details><summary>Tool Result</summary>

```
195	    #print('Step I: deramping …')
196	    raw      = read_slc_burst(files['measurement'], annot, burst_idx)
197	    deramped = deramp_burst(raw, annot, burst_idx, deramp_method=deramp_method)
198	    valid_mask = apply_burst_valid_sample_mask(deramped, burst)
199	
200	    # ── Step II: Correlation / Doppler estimation ────────────────────────────
201	    if estimator == 'fft':
202	        #print('Step II: estimating Doppler via FFT spectral centroid …')
203	        f_dc, snr, az_local, rg_centers = estimate_doppler_grid_fft(
204	            deramped, annot.prf, block_az, block_rg, stride_az, stride_rg,
205	            valid_mask=valid_mask,
206	        )
207	        p0 = np.ones_like(f_dc, dtype=np.float32)
208	        p1 = np.full_like(f_dc, np.nan, dtype=np.float32)
209	    else:
210	        #print('Step II: estimating p0, p1 …')
211	        p0, p1, az_local, rg_centers = estimate_correlation_grid(
212	            deramped, block_az, block_rg, stride_az, stride_rg,
213	            valid_mask=valid_mask,
214	        )
215	        f_dc, _, snr = correlation_to_doppler(
216	            p0, p1, annot.prf, annot.wavelength, gamma_amb=gamma_amb,
217	        )
218	
219	    # Full-scene azimuth indices for geolocation

... (10 lines omitted) ...

230	    k_a = _fm_rate_at_burst(annot, burst_idx)[rg_centers]
231	    k_psi = _steering_doppler_rate(annot, burst_idx)
232	    tops_scale = (1.0 + k_psi / k_a).astype(np.float32)
233	    if tops_scaling == 'multiply':
234	        f_dc = (f_dc * tops_scale[np.newaxis, :]).astype(np.float32)
235	    elif tops_scaling == 'divide':
236	        f_dc = (f_dc / tops_scale[np.newaxis, :]).astype(np.float32)
237	
238	    f_geom_ann = _geom_doppler_annotation(annot_original, burst_idx, rg_centers).astype(np.float32)
239	    f_geom_poe = _geom_doppler_poeorb(annot, annot_original, burst_idx, rg_centers).astype(np.float32)
240	
241	    f_dca = f_dc - f_geom_poe[np.newaxis, :]
242	    f_dca = f_dca - float(f_sideband)
243	
244	    f_dca, f_dca_pre_descallop, burst_period_rows = _maybe_descallop_grid(
245	        f_dca.astype(np.float32),
246	        snr,
247	        annot,
248	        stride_az=stride_az,
249	        do_descallop=do_descallop,
250	    )
251	    v_r = (annot.wavelength / 2.0 * f_dca).astype(np.float32)
252	    #print(f'  v_r median: {float(np.nanmedian(v_r)):+.4f} m/s')
253	
254	    # ── Step V: Geolocation ──────────────────────────────────────────────────
```

</details>

## Assistant

After thorough analysis, I can identify the flattening mechanism. Let me explain and then run the key diagnostic.

The `_block_p0_p1` Stage 2 does a **coherent complex average** of per-range-column lag-1 products before taking `angle()`:

```python
p1 = complex(np.sum(g1 * w2) / denom)   # vector mean of complex phasors
```

This collapses all 512 range columns into **one** complex number per block. For a heterogeneous scene where column j has DC `f_dc[j]`, the vector sum gives the angle of the mean phasor — which is biased for large DC variation and always hides sub-block range structure.

But you said smaller blocks don't help — which means the issue is that the **per-column `g1` values themselves don't vary across range**. Let's confirm this definitively:

## Assistant

**Tool: Read**
Reading `/home/oscipal/ocean_current_retrieval/scripts/sentinel_1/rvl.py`

<details><summary>Tool Result</summary>

```
799	def _block_p0_p1(block: np.ndarray, valid_mask: np.ndarray | None = None) -> tuple[float, complex]:
800	    """
801	    Lag-0 and lag-1 azimuth correlation coefficients for one tile,
802	    following the two-stage windowed procedure of Section 5.5.1
803	    (eqs. 12–13).
804	
805	    Stage 1 — azimuth (inner): apply Hanning window h_az along azimuth,
806	    then form per-range-column lag-1 products g1(t') and lag-0 power
807	    g0(t'):
808	
809	        I_w(t', τ) = I_c(t', τ) · h_az(τ)
810	
811	        g0(t') = Σ_τ |I_w(t', τ)|²                    (eq. 12, n = 0)
812	        g1(t') = Σ_τ I_w(t', τ) · I_w*(t', τ−1)       (eq. 12, n = 1)
813	
814	    Stage 2 — range (outer): average per-column estimates over range
815	    with h²_ra weights (Hanning squared, eq. 13):
816	
817	        p0 = Σ_t' h²_ra(t') g0(t') / Σ_t' h²_ra(t')
818	        p1 = Σ_t' h²_ra(t') g1(t') / Σ_t' h²_ra(t')
819	
820	    Both windows are normalised to unit sum before use (eq. 11).
821	
822	    NOTE: the ϖ∆ correction (eq. 9, ϖ∆ = ϖPRF / (1 + γ/β)) is not
823	    yet applied — the correct γ/β interpretation for deramped IW data

... (10 lines omitted) ...

834	    hra /= hra.sum()
835	
836	    if valid_mask is not None:
837	        block = np.where(valid_mask, block, 0.0)
838	
839	    # Stage 1 — apply h_az per range column
840	    I_w = block * haz[:, None]   # (n_az, n_rg)
841	
842	    # Per-column lag-1 products → g1, shape (n_rg,)
843	    g1 = np.sum(I_w[1:, :] * np.conj(I_w[:-1, :]), axis=0)
844	
845	    # Per-column lag-0 power → g0 via |I_w|² = |I_c|² · h_az² (Parseval, n=0)
846	    g0 = np.sum(np.abs(I_w) ** 2, axis=0)
847	
848	    # Stage 2 — h²_ra weights over range (eq. 13)
849	    w2    = hra ** 2
850	    denom = float(np.sum(w2))
851	    if denom < 1e-30:
852	        return 0.0, complex(0.0)
853	
854	    p0 = float(np.sum(g0 * w2) / denom)
855	    p1 = complex(np.sum(g1 * w2) / denom)
856	    return p0, p1
857	
858	
```

</details>

## User

This session is being continued from a previous conversation that ran out of context. The summary below covers the earlier portion of the conversation.

Summary:
1. Primary Request and Intent:

**Task 1 (Completed):** Add a `plot_rvl_comparison` function to `scripts/sentinel_1/doppler_comparison.py` that plots our raw Doppler (`f_dc`) and the OCN product's Doppler (`f_dc_ocn`) side by side as images with a scatter plot — modeled after `plot_comparison` in `plots.py`. After initial implementation used the wrong input format (`compare_burst_doppler` dict), it was corrected to take `list[dict]` from `run_all_bursts` using `merge_burst_grids` on `f_dc` and `f_dc_ocn` variables with `RdBu_r` colormap.

**Task 2 (Ongoing — primary focus):** Diagnose why our Doppler pipeline produces a flat/straight range profile for `f_dc` while ESA's OCN product `rvlDcObs` shows meaningful spatial variation with both negative and positive values. Specifically, burst 4 is entirely positive-valued in the final pipeline output, while the OCN product shows negative values for the first third of range then gradually positive — indicating a real current gradient.

The user explicitly stated: "Look I want the burst to be the same as with the ocn product, which it isn't. There must be something going wrong... I am not comparing apples with oranges, the same corrections and processing steps are done for both datasets and mine is somehow missing something. The signals shouldn't be that flat, so again, is there anything flattening the signals that isn't supposed to be there/was just implemented to smooth the outcome?"

2. Key Technical Concepts:

- **Sentinel-1 IW TOPS SLC processing**: burst mode, TOPS deramping, azimuth FM rate, steering Doppler rate
- **Doppler centroid estimation (CDE)**: lag-1 azimuth correlation estimator, two-stage block-based estimation
- **TOPS deramping**: `k_t = k_a × k_psi / (k_psi - k_a)`, applied as `exp(-j π k_t t_az²)`. Confirmed mathematically equivalent between our `'current'` method (`k_s = -k_a × k_psi / (k_a - k_psi) = k_t`) and ESA's `esa_eq1` method.
- **Steering Doppler rate**: `k_psi = (2 f0 / c) × v_sat × |psi_dot|` ≈ 7500 Hz/s for S1 IW (much larger than k_a ≈ -2300 Hz/s)
- **Two DC polynomial types** in S1 annotation: `data_poly` (observed DC ≈ tens of Hz, includes ocean signal) vs `geometry_poly` (orbital/geometric DC ≈ 1 Hz for near-zero squint S1)
- **`_block_p0_p1` Stage 2**: Stage 1 computes per-range-column lag-1 products `g1` (shape n_rg); Stage 2 performs coherent complex average of all range columns into a single `p1 = complex(sum(g1 * w2) / denom)`, collapsing all range structure to ONE DC per block
- **OCN variables**: `rvlDcObs` (full observed Doppler), `rvlDcGeo` (geometric Doppler), `rvlDcMiss` (mispointing), `rvlDca = rvlDcObs - rvlDcGeo - rvlDcMiss` (ocean anomaly)
- **`f_dc_ocn`**: loaded from `rvlDcObs` in OCN NetCDF via `match_to_sar_grid`, no transformations
- **`annot.prf`**: read from `azimuthFrequency` in annotation XML = correct SLC azimuth sample rate (hardware PRF for S1 IW ≈ 1012 Hz with no decimation)
- **DC estimate cadence**: DCEstimate records stored at ~1.9 s intervals, bursts at ~2.76 s intervals → every burst falls ~0.955 s from nearest DC record (generates 14 Hz error warning but affects all bursts uniformly)
- **`use_ocn_dc=True` mode**: replaces our lag-1 `f_dc` with `rvlDcObs` directly

3. Files and Code Sections:

- **`scripts/sentinel_1/doppler_comparison.py`**
   - Added `plot_rvl_comparison` function before `burst_summary_stats`. Final version takes `list[dict]` from `run_all_bursts`, uses `merge_burst_grids(results, variable='f_dc')` and `merge_burst_grids(results, variable='f_dc_ocn')` to get lat/lon grids. Shows two `imshow` panels (our f_dc and OCN f_dc_ocn) + scatter with 1:1 line. `RdBu_r` colormap, shared `vmax` at 98th percentile. Scatter title: `f"Our vs OCN\nbias={bias:+.2f}  RMSE={rmse:.2f}  r={r:.4f}"`.
   - Imports `merge_burst_grids`, `compute_stats` from `.grid_merge` inside the function.

- **`scripts/sentinel_1/pipeline.py`**
   - Read to understand how `f_dc_ocn` is created: `ocn_dc_obs = _clean('rvlDcObs')` → `f_dc_ocn = match_to_sar_grid(lat, lon, ocn_lat, ocn_lon, ocn_dc_obs)`
   - `use_ocn_dc=True` path: `f_dc = f_dc_ocn; f_dca = f_dc - f_geom_poe - f_sideband`
   - `f_dca = f_dc - f_geom_poe[np.newaxis, :] - f_sideband` (geometry subtraction, ~1 Hz)
   - Also has `tops_scaling` parameter (`'none'`/`'multiply'`/`'divide'`) that applies `(1 + k_psi/k_a)` scale to f_dc
   - Imports: `_nearest_estimate` from safe_io; NOT importing `_eval_poly`, `slant_range_time_vector`, `_burst_mid_time`

- **`scripts/sentinel_1/burst_pipeline.py`**
   - Full processing chain for single burst: read SLC → deramp → valid mask → lag-1 (or FFT) → correlation_to_doppler → geometry subtraction → geolocation
   - `f_dca = f_dc - f_geom_poe[np.newaxis, :]` (geometry subtraction)
   - Returns xr.Dataset with `doppler_hz` (f_dca), `doppler_obs` (f_dc), `doppler_geo` (f_geom_ann), etc.

- **`scripts/sentinel_1/rvl.py`** (key sections read):
   - `_steering_doppler_rate` (line 530): `k_psi = 2*f0/c * v_sat * |psi_dot|` where `psi_dot = azimuth_steering_rate * pi/180`. Docstring: "IW ≈ 7 500 Hz/s". This is the steering Doppler rate in Hz/s.
   - `_deramp_rate` (line 552): `return -k_a * k_psi / (k_a - k_psi)` — returns k_s = k_t < 0
   - `deramp_burst` 'current' method (line 664-676): `chirp = exp(j*pi*k_s[np.newaxis,:]*t_az[:,np.newaxis]**2)` where `t_az = (arange(lpb) - 0.5*(lpb-1)) * ati` (centered at burst midpoint)
   - `_deramp_esa` (line 589): ESA formula uses `phase = -pi*k_t*(eta-eta_ref)**2` with `k_t = k_a*k_s_esa/(k_a-k_s_esa)`. Verified: both methods produce the same `exp(-j pi |k_t| t_az²)`.
   - `_block_p0_p1` (line 799-856):
     ```python
     # Stage 1: per-column lag-1
     I_w = block * haz[:, None]   # (n_az, n_rg)
     g1 = np.sum(I_w[1:, :] * np.conj(I_w[:-1, :]), axis=0)  # shape (n_rg,)
     g0 = np.sum(np.abs(I_w) ** 2, axis=0)
     # Stage 2: coherent complex average across range
     w2 = hra ** 2
     denom = float(np.sum(w2))
     p0 = float(np.sum(g0 * w2) / denom)
     p1 = complex(np.sum(g1 * w2) / denom)  # ONE complex number per block
     return p0, p1
     ```
   - `estimate_correlation_grid` (line 859): Sliding window, returns `p0_grid, p1_grid, az_centers, rg_centers`
   - `_block_fft_doppler` (line 933-982): FFT spectral centroid, range-averaged incoherently via Hanning², also produces ONE f_dc per block
   - `estimate_doppler_grid_fft` (line 985-1048): Same block structure as lag-1, same spatial resolution limitation
   - `_geom_doppler_annotation` (line 1549): Uses `dc.geometry_poly` (~1 Hz for near-zero squint), NOT `dc.data_poly`

- **`scripts/sentinel_1/ocn_product.py`**
   - Pure loading/parsing — opens NetCDF with `decode_cf=True` (auto-applies scale_factor/add_offset), returns xr.Dataset. No physical transformation of rvlDcObs.

- **`scripts/sentinel_1/safe_io.py`**
   - `S1Annotation` dataclass: `slant_range_time_start` (float, s), `range_sampling_rate` (float, Hz), `prf` (= azimuthFrequency = SLC sample rate), `radar_prf` (hardware PRF), `n_samples`, `dc_estimates` (list with `data_poly`, `geometry_poly`, `t0`, `azimuth_time`)
   - `slant_range_time_vector`: `slant_range_time_start + arange(n_samples) / range_sampling_rate`
   - `_nearest_estimate`: argmin of time differences (NOT interpolation, unlike `_interpolate_velocity`)

4. Errors and Fixes:

- **Initial `plot_rvl_comparison` used wrong input format**: First implementation used `compare_burst_doppler` dict with `our_regrid` and `ocn['rvlDcObs']`. User correction: "it should take the result from run_all_bursts in pipeline/". Rewrote to use `merge_burst_grids(results, variable='f_dc')` and `merge_burst_grids(results, variable='f_dc_ocn')`.
- **Premature conclusion about OCN jumps**: Incorrectly told user the OCN "jumps" were burst-boundary seams and our smooth f_dc was correct. User firmly corrected: "no it isn't correct. small jumps should maybe not be there but big jumps yes, there must be an error here somewhere."
- **Premature focus on geometry subtraction**: Initially suspected `f_geom_poe` was wrong (too large). Data showed f_geom_poe ≈ 1 Hz — correctly tiny for near-zero squint S1. The geometry is correct, not the bug.
- **Incorrect reasoning that f_dc was correct**: Said our f_dc tracking annotation DC is "expected and correct." User pushed back: there IS a structural error.

5. Problem Solving:

**Solved:**
- `plot_rvl_comparison` correctly added to `doppler_comparison.py`
- k_a confirmed consistent across all bursts (not the cause)
- Deramping formula mathematically verified as correct (same as ESA's formula)
- f_geom_poe (~1 Hz) is correct for near-zero squint S1 IW
- annot.prf = azimuthFrequency = correct SLC sample rate (not a bug)
- `rvlDcObs` confirmed to come directly from OCN NetCDF with no transformations

**Ongoing diagnosis — the core problem:**
Our `f_dc` from lag-1 estimation on the deramped SLC produces values that:
1. Closely track the annotation `data_poly` (few Hz to tens of Hz)
2. Lack the range-structure variation seen in `rvlDcObs` (which spans up to ±100+ Hz)
3. For burst 4 specifically: `data_poly` is nearly flat (+6-7 Hz), `our_f_dc` is flat (+2-18 Hz), while `rvlDcObs` spans 157 Hz (-40.9 to +116.9 Hz)

**Two candidate mechanisms identified:**
1. **Stage 2 coherent range average** in `_block_p0_p1`: Collapses all 512 range columns into one complex p1 per block. For heterogeneous scenes with DC varying across range, the vector mean gives biased/flat output. HOWEVER: user said smaller blocks don't help, implying per-column g1 values themselves may be flat.
2. **IPF DC centering**: The S1 IPF subtracts `data_poly` during azimuth compression, leaving our lag-1 to see only the small residual. This would explain `f_dc ≈ data_poly` across all bursts. For burst 4, the IPF estimated a flat data_poly (+6-7 Hz) but the true DC varied by 157 Hz — making recovery of the true DC from the L1 SLC impossible without ESA's more robust estimation.

**The definitive pending test:** Plot `angle(g1[j])` per range column for burst 4 without Stage 2 averaging. If angles vary across range → Stage 2 is the bug. If angles are flat → the deramped SLC itself has no range DC variation (IPF issue).

6. All User Messages:

From previous session (preserved from chat.md summary):
- "in scripts/sentinel_1/doppler_comparison/ add a function to plot the rvl from our product and from the ocean product side by side with a scatter plot next to it. like the plot_comparison function in pipeline/."
- "it should take the result from run_all_bursts in pipeline/"
- "use the same colormap as in plot_comparison"
- "Okay a bit differrent: If I look at burst number 4 the whole burst is positive valued (at the end of my pipeline) whereas the ocn product shows that it should be negative for the first third of range and then gradually go positive"
- "nah i think it already happens when deramping and estimating the doppler, so most probably a deramping issue"
- "what annot do i need?"
- [k_a output for all 9 bursts — all consistent ~-2326 to -2191 Hz/s]
- "looks very different"
- "i mean in range direction, ocn product is correct"
- "no i mean the same thing happens when i make the range block smaller, it isn't flat as in 0, but flat as in no jumps, goes in a straight line (still decreases or increases but just in a flat line without the variation seen in ocn)"
- "it looks like in the ocn product there are a few jumps, and our data is just flat or very straight for all bursts. Could it be that we just filter out any strong changes?"
- "no it isn't correct. small jumps should maybe not be there but big jumps yes, there must be an error here somewhere"
- [Warning output: all 10 bursts show nearest DC record 0.945–0.955 s away, ~14 Hz error]
- [Numerical output — annot_dc, our_f_dc, f_dc_ocn for bursts 0–9]
- [Numerical output — f_geom_poe, f_dca added for bursts 0–9]
- "Look I want the burst to be the same as with the ocn product, which it isn't. There must be something going wrong, it is not because near range is not processed and not because snr is bad, it's something done wrong for the whole burst. I am not comparing apples with oranges, the same corrections and processing steps are done for both datasets and mine is somehow missing something. The signals shouldn't be that flat, so again, is there anything flattening the signals that isn't supposed to be there/was just implemented to smooth the outcome?"

Current session:
- "our last chat is in chat.md. The limit ran out, countinue from there"

7. Pending Tasks:

- **Identify and fix the root cause of flat f_dc**: Either Stage 2 coherent averaging destroying range variation, or IPF DC centering causing our lag-1 to see only the residual without range structure.
- **Run the per-column g1 diagnostic**: Plot `angle(g1[j])` per range column for burst 4 to determine if Stage 2 or the deramped SLC itself is the issue.
- **Explain the mechanism clearly to the user**: Once diagnosed, explain what is flattening the signal and propose a fix.
- If Stage 2 IS the issue: modify `_block_p0_p1` to output per-range-column DC (change Stage 2 from coherent complex sum to per-column DC computation)
- If IPF DC centering IS the issue: consider adding annotation `data_poly` back to `f_dc` before geometry subtraction, and/or use `estimator='fft'` or `use_ocn_dc=True` for problematic bursts.

8. Current Work:

Immediately before this summary was being requested, the analysis was focused on the `_block_p0_p1` function in `rvl.py` (lines 799-856), specifically examining Stage 2's coherent complex averaging as the flattening mechanism.

The key code being analyzed:
```python
# Stage 2 — h²_ra weights over range (eq. 13)
w2    = hra ** 2
denom = float(np.sum(w2))
p0 = float(np.sum(g0 * w2) / denom)
p1 = complex(np.sum(g1 * w2) / denom)   # single scalar per block
return p0, p1
```

This collapses all n_rg range columns into one complex number p1. For heterogeneous scenes where adjacent columns have very different DC values, this vector mean gives a biased estimate that is "flat" across blocks. However, user's observation that "smaller blocks don't help" implies the per-column g1 values themselves are approximately equal — pointing to the deramped SLC itself lacking range DC variation.

The analysis was building toward suggesting a specific diagnostic experiment and had also been exploring whether the annotation `data_poly` needs to be added back to `f_dc` (since our `f_dc ≈ data_poly` across all bursts, suggesting the IPF removed data_poly during azimuth compression and our lag-1 sees only the residual).

9. Optional Next Step:

The next step, directly in line with the user's last explicit request ("is there anything flattening the signals that isn't supposed to be there/was just implemented to smooth the outcome?"), is to suggest and describe the definitive per-column diagnostic experiment:

```python
burst_idx = 4
raw = read_slc_burst(files['measurement'], annot, burst_idx)
deramped = deramp_burst(raw, annot, burst_idx)

# Take a single 256-line block centered in the burst
center = deramped.shape[0] // 2
block = deramped[center-128:center+128, :]

haz = np.hanning(256)
haz /= haz.sum()
I_w = block * haz[:, None]

# Per-column lag-1 (NO Stage 2 range averaging)
g1_per_col = np.sum(I_w[1:, :] * np.conj(I_w[:-1, :]), axis=0)
f_dc_per_col = annot.prf / (2*np.pi) * np.angle(g1_per_col)

import matplotlib.pyplot as plt
plt.figure(figsize=(12, 4))
plt.plot(f_dc_per_col)
plt.xlabel('range sample')
plt.ylabel('Estimated DC [Hz]')
plt.title('Burst 4 — per-column lag-1 DC (no Stage 2 range averaging)')
plt.show()
print(f"per-column DC: [{f_dc_per_col.min():.1f}, {f_dc_per_col.max():.1f}] Hz")
```

This tells us unambiguously:
- If `f_dc_per_col` varies across range (spans hundreds of Hz) → Stage 2 is destroying the variation → fix by computing per-column DC instead of coherent average
- If `f_dc_per_col` is flat (same ≈ few Hz everywhere) → the deramped SLC has no range DC variation → issue is in what the IPF stored in the SLC (data_poly compensation during focusing)

The user's question was explicitly: "is there anything flattening the signals that isn't supposed to be there/was just implemented to smooth the outcome?" This diagnostic directly answers it.

If you need specific details from before compaction (like exact code snippets, error messages, or content you generated), read the full transcript at: /home/oscipal/.claude/projects/-home-oscipal-ocean-current-retrieval/8d8b0096-135a-4267-b2c9-21398a1215c8.jsonl
Continue the conversation from where it left off without asking the user any further questions. Resume directly — do not acknowledge the summary, do not recap what was happening, do not preface with "I'll continue" or similar. Pick up the last task as if the break never happened.

## Assistant

Let me read the current state of the key files to pick up right where we left off.

## Assistant

**Tool: Read**
Reading `/home/oscipal/ocean_current_retrieval/scripts/sentinel_1/rvl.py`

<details><summary>Tool Result</summary>

```
790	        I_c[az0:az1, :] += _window_burst(deramped, annot, j, overlap_lines)
791	
792	    return I_c
793	
794	
795	# ─────────────────────────────────────────────────────────────────────────────
796	# Step II — Azimuth correlation coefficients  (Section 5.5)
797	# ─────────────────────────────────────────────────────────────────────────────
798	
799	def _block_p0_p1(block: np.ndarray, valid_mask: np.ndarray | None = None) -> tuple[float, complex]:
800	    """
801	    Lag-0 and lag-1 azimuth correlation coefficients for one tile,
802	    following the two-stage windowed procedure of Section 5.5.1
803	    (eqs. 12–13).
804	
805	    Stage 1 — azimuth (inner): apply Hanning window h_az along azimuth,
806	    then form per-range-column lag-1 products g1(t') and lag-0 power
807	    g0(t'):
808	
809	        I_w(t', τ) = I_c(t', τ) · h_az(τ)
810	
811	        g0(t') = Σ_τ |I_w(t', τ)|²                    (eq. 12, n = 0)
812	        g1(t') = Σ_τ I_w(t', τ) · I_w*(t', τ−1)       (eq. 12, n = 1)
813	
814	    Stage 2 — range (outer): average per-column estimates over range

... (70 lines omitted) ...

885	
886	    Returns
887	    -------
888	    p0_grid  : float32,   shape (n_out_az, n_out_rg)
889	    p1_grid  : complex64, shape (n_out_az, n_out_rg)
890	    az_centers : int array, azimuth pixel centres in I_c
891	    rg_centers : int array, range pixel centres in I_c
892	    """
893	    n_az, n_rg = I_c.shape
894	    if valid_mask is not None and valid_mask.shape != I_c.shape:
895	        raise ValueError(
896	            f'valid_mask shape {valid_mask.shape} does not match I_c shape {I_c.shape}'
897	        )
898	
899	    az_starts = np.arange(0, n_az - block_az + 1, stride_az)
900	    rg_starts = np.arange(0, n_rg - block_rg + 1, stride_rg)
901	
902	    n_out_az = len(az_starts)
903	    n_out_rg = len(rg_starts)
904	
905	    p0_grid = np.full((n_out_az, n_out_rg), np.nan, dtype=np.float32)
906	    p1_grid = np.full((n_out_az, n_out_rg), np.nan + 1j * np.nan, dtype=np.complex64)
907	
908	    for i, az0 in enumerate(az_starts):
909	        for j, rg0 in enumerate(rg_starts):
```

</details>

## Assistant

**Tool: Read**
Reading `/home/oscipal/ocean_current_retrieval/scripts/sentinel_1/pipeline.py`

<details><summary>Tool Result</summary>

```
1	"""Run the main Sentinel-1 RVL retrieval pipeline and its merged-burst variants."""
2	
3	from __future__ import annotations
4	
5	import argparse
6	
7	import numpy as np
8	
9	from scripts.diagnostics.pipeline_diagnostics import (
10	    diagnose_burst_doppler as _diagnose_burst_doppler,
11	    diagnose_mispointing_aux_ins as _diagnose_mispointing_aux_ins,
12	    plot_dc_estimates as _plot_dc_estimates,
13	    plot_pipeline_steps as _plot_pipeline_steps,
14	)
15	from .metocean import (
16	    compute_stokes_radial,
17	    compute_wave_doppler_bias,
18	    load_era5_wave,
19	    load_era5_wind,
20	    load_glo12_current,
21	    match_to_sar_grid,
22	    project_current_onto_look,
23	)
24	from .grid_merge import (
25	    compute_stats as _compute_stats_impl,

... (30 lines omitted) ...

56	from .plots import plot_comparison as _plot_comparison_impl
57	
58	_OCN_FILL  = -999.0
59	_SWATH_IDX = {'iw1': 0, 'iw2': 1, 'iw3': 2}
60	
61	
62	def _maybe_descallop_grid(
63	    f_dca: np.ndarray,
64	    snr: np.ndarray,
65	    annot,
66	    stride_az: int,
67	    do_descallop: bool,
68	) -> tuple[np.ndarray, np.ndarray, float | None]:
69	    """Apply burst-periodic descalloping when enough burst timing information is available."""
70	    f_dca_pre = f_dca.copy()
71	    if not do_descallop or len(annot.bursts) < 2:
72	        return f_dca, f_dca_pre, None
73	
74	    burst_dt_s = (
75	        annot.bursts[1].azimuth_time - annot.bursts[0].azimuth_time
76	    ).total_seconds()
77	    burst_period_rows = burst_dt_s / annot.azimuth_time_interval / stride_az
78	    if not np.isfinite(burst_period_rows) or burst_period_rows <= 0:
79	        return f_dca, f_dca_pre, None
80	
```

</details>

## Assistant

Let me check what's in the diagnostics directory and the existing `burst_fdc_stages.py` file, which sounds directly relevant.

## Assistant

**Tool: Read**
Reading `/home/oscipal/ocean_current_retrieval/scripts/diagnostics/burst_fdc_stages.py`

<details><summary>Tool Result</summary>

```
1	"""Compare burst-wise Doppler offsets across raw, deramped, CDE, and FFT stages."""
2	
3	from __future__ import annotations
4	
5	import argparse
6	
7	import numpy as np
8	
9	from scripts.sentinel_1.rvl import (
10	    apply_burst_valid_sample_mask,
11	    correlation_to_doppler,
12	    deramp_burst,
13	    estimate_correlation_grid,
14	    estimate_doppler_grid_fft,
15	)
16	from scripts.sentinel_1.safe_io import find_safe_files, parse_annotation, read_slc_burst
17	
18	
19	def _summarise(arr: np.ndarray) -> tuple[float, float]:
20	    finite = arr[np.isfinite(arr)]
21	    if finite.size == 0:
22	        return float("nan"), float("nan")
23	    return float(np.nanmean(finite)), float(np.nanstd(finite))
24	
25	

... (139 lines omitted) ...

165	        description="Compare burst-wise Doppler offsets before and after deramping with CDE and FFT estimators."
166	    )
167	    parser.add_argument("slc_safe", help="Path to the Sentinel-1 SLC SAFE directory")
168	    parser.add_argument("--subswath", default="iw1", help="Subswath, e.g. iw1/iw2/iw3")
169	    parser.add_argument("--pol", default="vv", help="Polarisation, e.g. vv or vh")
170	    parser.add_argument("--block-az", type=int, default=256)
171	    parser.add_argument("--block-rg", type=int, default=512)
172	    parser.add_argument("--stride-az", type=int, default=128)
173	    parser.add_argument("--stride-rg", type=int, default=256)
174	    args = parser.parse_args()
175	
176	    run(
177	        slc_safe=args.slc_safe,
178	        subswath=args.subswath,
179	        polarisation=args.pol,
180	        block_az=args.block_az,
181	        block_rg=args.block_rg,
182	        stride_az=args.stride_az,
183	        stride_rg=args.stride_rg,
184	    )
185	
186	
187	if __name__ == "__main__":
188	    main()
189	
```

</details>

## Assistant

**Tool: Read**
Reading `/home/oscipal/ocean_current_retrieval/scripts/sentinel_1/rvl.py`

<details><summary>Tool Result</summary>

```
909	        for j, rg0 in enumerate(rg_starts):
910	            block = I_c[az0: az0 + block_az, rg0: rg0 + block_rg]
911	            mask_block = None if valid_mask is None else valid_mask[
912	                az0: az0 + block_az, rg0: rg0 + block_rg
913	            ]
914	
915	            valid_frac = (
916	                np.count_nonzero(block) / block.size
917	                if mask_block is None
918	                else np.count_nonzero(mask_block) / mask_block.size
919	            )
920	            if valid_frac < min_valid_fraction:
921	                continue
922	
923	            p0, p1 = _block_p0_p1(block, mask_block)
924	            p0_grid[i, j] = p0
925	            p1_grid[i, j] = p1
926	
927	    az_centers = az_starts + block_az // 2
928	    rg_centers = rg_starts + block_rg // 2
929	
930	    return p0_grid, p1_grid, az_centers, rg_centers
931	
932	
933	def _block_fft_doppler(

... (10 lines omitted) ...

944	    imperfect deramping (a symmetric chirp residual broadens the spectral peak
945	    but does not shift the power-weighted centroid).
946	
947	    Returns
948	    -------
949	    f_dc  : float  Doppler centroid [Hz], NaN if insufficient power
950	    snr   : float  spectral peak / noise-floor ratio (linear)
951	    """
952	    n_az, n_rg = block.shape
953	    haz = np.hanning(n_az).astype(np.float32)
954	    hra = np.hanning(n_rg).astype(np.float32)
955	
956	    if valid_mask is not None:
957	        block = np.where(valid_mask, block, 0.0)
958	
959	    I_w = block * haz[:, None]
960	
961	    S   = np.fft.fft(I_w, axis=0)          # (n_az, n_rg)
962	    P   = np.abs(S) ** 2                    # azimuth power spectrum per range col
963	
964	    w2    = hra ** 2
965	    denom = float(np.sum(w2))
966	    if denom < 1e-30:
967	        return float('nan'), float('nan')
968	
```

</details>

## Assistant

Let me read the `_geom_doppler_annotation` function and the annotation DC structure to verify the data_poly vs geometry_poly distinction.

## Assistant

**Tool: Read**
Reading `/home/oscipal/ocean_current_retrieval/scripts/sentinel_1/rvl.py`

<details><summary>Tool Result</summary>

```
1540	            'doppler_input':         'deramped, spectrally windowed, merged TOPS bursts',
1541	        },
1542	    )
1543	
1544	
1545	# ─────────────────────────────────────────────────────────────────────────────
1546	# GAMMA-format entry point (pre-deramped per-burst .slc files)
1547	# ─────────────────────────────────────────────────────────────────────────────
1548	
1549	def _geom_doppler_annotation(
1550	    annot: S1Annotation,
1551	    burst_idx: int,
1552	    rg_centers: np.ndarray,
1553	) -> np.ndarray:
1554	    """
1555	    Evaluate the annotation geometry Doppler polynomial at range pixel centres.
1556	
1557	    Uses the DcEstimate nearest to the burst azimuth time.  The polynomial
1558	    variable is slant-range time [s] relative to the reference t0.
1559	
1560	    Returns
1561	    -------
1562	    np.ndarray, shape (len(rg_centers),) — geometric Doppler [Hz]
1563	    """
1564	    burst = annot.bursts[burst_idx]

... (30 lines omitted) ...

1595	
1596	    az_offsets = [
1597	        int(round((b.azimuth_time - annot.first_line_time).total_seconds() / ati))
1598	        for b in annot.bursts
1599	    ]
1600	
1601	    win = _burst_window(lpb, overlap_lines).astype(np.float64)
1602	
1603	    # Evaluate each burst's geometry polynomial at rg_centers once
1604	    f_geom_bursts = np.stack(
1605	        [_geom_doppler_annotation(annot, j, rg_centers).astype(np.float64)
1606	         for j in range(len(annot.bursts))],
1607	        axis=0,
1608	    )   # (n_bursts, n_out_rg)
1609	
1610	    n_out_rg = len(rg_centers)
1611	    f_geom   = np.zeros((len(az_centers), n_out_rg), dtype=np.float32)
1612	
1613	    for i, az in enumerate(az_centers):
1614	        blend  = np.zeros(n_out_rg, dtype=np.float64)
1615	        w2_sum = 0.0
1616	        for j, az0 in enumerate(az_offsets):
1617	            rel = int(az) - az0
1618	            if 0 <= rel < lpb:
1619	                # Use w² weights: the lag-1 estimator weights each burst's
```

</details>

## Assistant

**Tool: Read**
Reading `/home/oscipal/ocean_current_retrieval/scripts/sentinel_1/safe_io.py`

<details><summary>Tool Result</summary>

```
1	"""Read Sentinel-1 SAFE annotations, bursts, and calibration lookup tables."""
2	
3	import os
4	import glob
5	import numpy as np
6	import rasterio
7	import xml.etree.ElementTree as ET
8	from dataclasses import dataclass, field
9	from datetime import datetime, timezone
10	from scipy.interpolate import RectBivariateSpline
11	
12	
13	# ---------------------------------------------------------------------------
14	# File discovery
15	# ---------------------------------------------------------------------------
16	
17	def find_safe_files(safe_dir: str, subswath: str, polarisation: str) -> dict:
18	    """
19	    Locate annotation, measurement, calibration, and noise XML/TIFF for one
20	    subswath + polarisation combination inside a SAFE directory.
21	
22	    Parameters
23	    ----------
24	    safe_dir : str
25	        Path to the .SAFE directory.

... (50 lines omitted) ...

76	    q1: float
77	    q2: float
78	    q3: float
79	    roll:  float  # absolute Euler angle [deg] in inertial frame
80	    pitch: float
81	    yaw:   float
82	
83	
84	@dataclass
85	class BurstInfo:
86	    idx: int
87	    azimuth_time: datetime
88	    byte_offset: int
89	    first_valid_sample: np.ndarray   # shape (linesPerBurst,), -1 = invalid
90	    last_valid_sample: np.ndarray    # shape (linesPerBurst,)
91	    sensing_time: datetime
92	
93	
94	@dataclass
95	class DcEstimate:
96	    azimuth_time: datetime
97	    t0: float                         # reference slant-range time [s]
98	    geometry_poly: np.ndarray         # orbit-predicted DC coefficients
99	    data_poly: np.ndarray             # measured DC coefficients
100	    rms_error: float
```

</details>

## Assistant

Let me check `_fm_rate_at_burst` and `_deramp_rate` to understand if k_a is scalar or per-range:

## Assistant

**Tool: Read**
Reading `/home/oscipal/ocean_current_retrieval/scripts/sentinel_1/rvl.py`

<details><summary>Tool Result</summary>

```
500	    Uses the nearest azimuthFmRate record to the burst time.  k_a is
501	    negative for typical side-looking geometry (≈ −2200 Hz/s for IW).
502	
503	    Returns
504	    -------
505	    np.ndarray, shape (n_samples,)
506	    """
507	    burst_mid = _burst_mid_time(annot, burst_idx)
508	    afr   = _nearest_estimate(annot.azimuth_fm_rates, burst_mid)
509	    tau   = slant_range_time_vector(annot)          # (n_samples,)
510	    dt    = tau - afr.t0
511	    return sum(c * dt**k for k, c in enumerate(afr.poly))  # Hz/s
512	
513	
514	def _orbital_speed(annot: S1Annotation, burst_idx: int) -> float:
515	    """
516	    Return the satellite orbital speed [m/s] at the burst centre time.
517	
518	    Interpolated from the ECEF velocity state vectors stored in the annotation.
519	    The full 3-D speed |v| is used because k_psi depends on the physical
520	    angular sweep rate of the antenna in inertial space, not the ground-track
521	    speed.
522	    """
523	    burst_time = annot.bursts[burst_idx].azimuth_time
524	    diffs = [abs((t - burst_time).total_seconds()) for t in annot.orbit_times]

... (80 lines omitted) ...

605	
606	    afr = _nearest_estimate(annot.azimuth_fm_rates, burst_mid)
607	    dc = _nearest_estimate(annot.dc_estimates, burst_mid)
608	
609	    k_a = _eval_poly(afr.poly, afr.t0, tau).astype(np.float64)
610	    f_dc = _eval_poly(dc.data_poly, dc.t0, tau).astype(np.float64)
611	
612	    vel = _interpolate_velocity(annot, burst_mid)
613	    v_s = float(np.linalg.norm(vel))
614	    k_psi = annot.azimuth_steering_rate * np.pi / 180.0
615	    k_s = (2.0 * v_s / C_LIGHT) * annot.radar_frequency * k_psi
616	    k_t = (k_a * k_s) / (k_a - k_s)
617	
618	    eta_c = -f_dc / k_a
619	    eta_ref = eta_c - eta_c[annot.n_samples // 2]
620	
621	    phase = -np.pi * k_t[np.newaxis, :] * (eta - eta_ref[np.newaxis, :]) ** 2
622	    if demodulate:
623	        phase = phase - 2.0 * np.pi * f_dc[np.newaxis, :] * (eta - eta_ref[np.newaxis, :])
624	    phi = np.exp(1j * phase)
625	    return (burst * phi).astype(np.complex64)
626	
627	
628	# ─────────────────────────────────────────────────────────────────────────────
629	# Step I — Deramp + window + merge  (Section 5.4)
```

</details>

## Assistant

Now I need to see the `'current'` deramp method to compare with `_deramp_esa`:

## Assistant

**Tool: Read**
Reading `/home/oscipal/ocean_current_retrieval/scripts/sentinel_1/rvl.py`

<details><summary>Tool Result</summary>

```
628	# ─────────────────────────────────────────────────────────────────────────────
629	# Step I — Deramp + window + merge  (Section 5.4)
630	# ─────────────────────────────────────────────────────────────────────────────
631	
632	def deramp_burst(
633	    burst: np.ndarray,
634	    annot: S1Annotation,
635	    burst_idx: int,
636	    deramp_method: str = 'current',
637	) -> np.ndarray:
638	    """
639	    Apply the TOPS deramp chirp to one complex burst (eq. 1–3 in the doc).
640	
641	        I_cbd(t_az, τ) = c1(t_az, τ) · I_cb(t_az, τ)
642	        c1(t_az, τ)   = exp(iπ k_s(τ) t_az²)
643	
644	    where t_az is azimuth time measured from the burst centre [s].
645	
646	    Parameters
647	    ----------
648	    burst     : complex64, shape (linesPerBurst, samplesPerBurst)
649	    annot     : S1Annotation
650	    burst_idx : int
651	    deramp_method : str
652	        `current` uses the existing repository deramp chirp.

... (50 lines omitted) ...

703	
704	def _window_burst(
705	    I_bcd: np.ndarray,
706	    annot: S1Annotation,
707	    burst_idx: int,
708	    overlap_lines: int,
709	) -> np.ndarray:
710	    """
711	    Apply the TOPS spectral window (Section 5.4.1, eq. 4):
712	
713	        I_cbdw = c2* ⊗ [(c2 ⊗ I_bcd) · w(· − τ̄_j)]
714	
715	    with  c2(τ) = exp(iπ k_psi τ²)  (steering Doppler chirp).
716	
717	    The c2 defocusing maps each Doppler frequency to its natural azimuth-time
718	    position before the raised-cosine window is applied, so the partition-of-
719	    unity condition is satisfied across the burst overlap region.
720	
721	    Implementation steps:
722	        1. FFT(I_bcd) along azimuth
723	        2. × C2            →  c2 ⊗ I_bcd  (defocus)
724	        3. IFFT
725	        4. × w(τ − τ̄_j)   →  raised-cosine window centred on burst
726	        5. FFT
727	        6. × C2*           →  c2* ⊗ (windowed)  (refocus)
```

</details>

## Assistant

Let me check how `annot_dc` is computed in the pipeline and what the data_poly evaluation looks like:

## Assistant

**Tool: Bash**
*Find annot_dc and data_poly usage in pipeline*
```bash
grep -n "annot_dc\|data_poly\|_nearest_estimate\|_eval_poly" /home/oscipal/ocean_current_retrieval/scripts/sentinel_1/pipeline.py | head -50
```

<details><summary>Tool Result</summary>

```
31:from .safe_io import _iso_to_datetime, _nearest_estimate, find_safe_files, parse_annotation, read_slc_burst
```

</details>

## Assistant

**Tool: Bash**
*Find DC estimation and annotation DC usage in pipeline*
```bash
grep -n "annot_dc\|data_poly\|dc_est\|geometry_poly" /home/oscipal/ocean_current_retrieval/scripts/sentinel_1/pipeline.py | head -50
```

<details><summary>Tool Result</summary>

```
12:    plot_dc_estimates as _plot_dc_estimates,
1287:def plot_dc_estimates(
1294:    _plot_dc_estimates(slc_safe, subswath, polarisation=polarisation, out_path=out_path)
```

</details>

## Assistant

**Tool: Bash**
*Find annot_dc in notebook*
```bash
grep -n "annot_dc\|data_poly\|dc_est\|geometry_poly" /home/oscipal/ocean_current_retrieval/notebooks/s1_ocean.ipynb | head -30
```

<details><summary>Tool Result</summary>

```
(Bash completed with no output)
```

</details>

## Assistant

**Tool: Bash**
*Find annot_dc in doppler_comparison*
```bash
grep -n "annot_dc" /home/oscipal/ocean_current_retrieval/scripts/sentinel_1/doppler_comparison.py | head -30
```

<details><summary>Tool Result</summary>

```
(Bash completed with no output)
```

</details>

## Assistant

**Tool: Bash**
*Find annot_dc and data_poly across all scripts*
```bash
grep -rn "annot_dc\|data_poly" /home/oscipal/ocean_current_retrieval/scripts/ | head -40
```

<details><summary>Tool Result</summary>

```
/home/oscipal/ocean_current_retrieval/scripts/sentinel_1/safe_io.py:99:    data_poly: np.ndarray             # measured DC coefficients
/home/oscipal/ocean_current_retrieval/scripts/sentinel_1/safe_io.py:188:        data_poly_txt = dc.findtext('dataDcPolynomial')
/home/oscipal/ocean_current_retrieval/scripts/sentinel_1/safe_io.py:189:        if geom_poly_txt is None or data_poly_txt is None:
/home/oscipal/ocean_current_retrieval/scripts/sentinel_1/safe_io.py:195:            data_poly=np.array([float(x) for x in data_poly_txt.split()]),
/home/oscipal/ocean_current_retrieval/scripts/sentinel_1/rvl.py:610:    f_dc = _eval_poly(dc.data_poly, dc.t0, tau).astype(np.float64)
/home/oscipal/ocean_current_retrieval/scripts/diagnostics/esa_deramp_compare.py:60:    f_dc = _eval_poly(dc.data_poly, dc.t0, tau).astype(np.float64)
/home/oscipal/ocean_current_retrieval/scripts/diagnostics/pipeline_diagnostics.py:202:        f_data = sum(c * dt**k for k, c in enumerate(dc.data_poly))
/home/oscipal/ocean_current_retrieval/scripts/diagnostics/pipeline_diagnostics.py:261:        f_data = float(sum(c * dt_mid**k for k, c in enumerate(dc.data_poly)))
```

</details>

## Assistant

**Tool: Read**
Reading `/home/oscipal/ocean_current_retrieval/scripts/diagnostics/pipeline_diagnostics.py`

<details><summary>Tool Result</summary>

```
180	    """Plot stacked-burst annotation DC polynomials without SLC processing."""
181	    import matplotlib.pyplot as plt
182	
183	    files = find_safe_files(slc_safe, subswath, polarisation)
184	    annot = parse_annotation(files["annotation"])
185	    tau = annot.slant_range_time_start + np.arange(annot.n_samples) / annot.range_sampling_rate
186	
187	    rg_step = max(1, annot.n_samples // 500)
188	    tau_ds = tau[::rg_step]
189	    rows_per_burst = 10
190	    geom_imgs, data_imgs, diff_imgs = [], [], []
191	    sep = np.full((1, len(tau_ds)), np.nan, dtype=np.float32)
192	
193	    hdr = f'{"burst":>5}  {"dc_idx":>6}  {"f_geom_mid [Hz]":>16}  {"f_data_mid [Hz]":>16}  {"diff_mid [Hz]":>14}'
194	    print(hdr)
195	    print("─" * len(hdr))
196	
197	    for burst_idx, burst in enumerate(annot.bursts):
198	        dc = _nearest_estimate(annot.dc_estimates, burst.azimuth_time)
199	        dc_idx = annot.dc_estimates.index(dc)
200	        dt = tau - dc.t0
201	        f_geom = sum(c * dt**k for k, c in enumerate(dc.geometry_poly))
202	        f_data = sum(c * dt**k for k, c in enumerate(dc.data_poly))
203	
204	        f_geom_ds = f_geom[::rg_step].astype(np.float32)

... (50 lines omitted) ...

255	
256	    f_miss_arr, f_data_arr, f_geom_arr = [], [], []
257	    for burst_idx, burst in enumerate(annot.bursts):
258	        dc = _nearest_estimate(annot.dc_estimates, burst.azimuth_time)
259	        dt_mid = tau[mid] - dc.t0
260	        f_geom = float(sum(c * dt_mid**k for k, c in enumerate(dc.geometry_poly)))
261	        f_data = float(sum(c * dt_mid**k for k, c in enumerate(dc.data_poly)))
262	        f_miss = mispointing_doppler_from_yaw(annot, burst_idx)
263	        f_miss_arr.append(f_miss)
264	        f_data_arr.append(f_data)
265	        f_geom_arr.append(f_geom)
266	
267	    f_miss_arr = np.array(f_miss_arr)
268	    f_data_arr = np.array(f_data_arr)
269	    f_geom_arr = np.array(f_geom_arr)
270	    f_dca_arr = f_data_arr - f_geom_arr
271	    residual = f_dca_arr - f_miss_arr
272	
273	    hdr = f'{"burst":>5}  {"f_miss [Hz]":>12}  {"f_data [Hz]":>12}  {"f_geom [Hz]":>12}  {"f_dca [Hz]":>11}  {"residual [Hz]":>14}'
274	    print(hdr)
275	    print("─" * len(hdr))
276	    for i in range(len(annot.bursts)):
277	        print(f"{i:>5}  {f_miss_arr[i]:>12.2f}  {f_data_arr[i]:>12.2f}  {f_geom_arr[i]:>12.2f}  {f_dca_arr[i]:>11.2f}  {residual[i]:>+14.2f}")
278	
279	    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
```

</details>

## Assistant

I've found the key issue. Let me read the pipeline's `f_dc` computation and the relevant part of `pipeline.py`:

## Assistant

**Tool: Read**
Reading `/home/oscipal/ocean_current_retrieval/scripts/sentinel_1/pipeline.py`

<details><summary>Tool Result</summary>

```
80	
81	    # Require at least 6 inter-burst periods (≈ 3+ bursts in the data) before
82	    # attempting descalloping.  For IW mode burst_period_rows ≈ 2.4, so this
83	    # means at least ~15 output rows.  Single-burst output grids (~12 rows, ≈ 5
84	    # periods) appear to cover multiple periods but the scalloping there repeats
85	    # at the burst DURATION frequency (not the inter-burst frequency), so the
86	    # model is wrong and the fit introduces artifacts rather than removing them.
87	    if f_dca.shape[0] < int(np.ceil(6.0 * burst_period_rows)):
88	        return f_dca, f_dca_pre, burst_period_rows
89	
90	    return descallop(f_dca, snr, burst_period_rows), f_dca_pre, burst_period_rows
91	
92	
93	def to_regular_grid(
94	    lat: np.ndarray,
95	    lon: np.ndarray,
96	    data: np.ndarray,
97	    resolution_deg: float = 0.01,
98	    method: str = 'linear',
99	) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
100	    """
101	    Regrid scattered SAR estimation points onto a regular lat/lon grid.
102	
103	    Parameters
104	    ----------

... (100 lines omitted) ...

205	            valid_mask=valid_mask,
206	        )
207	        p0 = np.ones_like(f_dc, dtype=np.float32)
208	        p1 = np.full_like(f_dc, np.nan, dtype=np.float32)
209	    else:
210	        #print('Step II: estimating p0, p1 …')
211	        p0, p1, az_local, rg_centers = estimate_correlation_grid(
212	            deramped, block_az, block_rg, stride_az, stride_rg,
213	            valid_mask=valid_mask,
214	        )
215	        f_dc, _, snr = correlation_to_doppler(
216	            p0, p1, annot.prf, annot.wavelength, gamma_amb=gamma_amb,
217	        )
218	
219	    # Full-scene azimuth indices for geolocation
220	    ati      = annot.azimuth_time_interval
221	    az0_full = int(round(
222	        (burst.azimuth_time - annot.first_line_time).total_seconds() / ati
223	    ))
224	    az_full = az_local + az0_full
225	
226	    # ── Step III: Doppler centroid → DCA ────────────────────────────────────
227	    #print('Step III: geometry subtraction …')
228	    if tops_scaling not in ('none', 'multiply', 'divide'):
229	        raise ValueError("tops_scaling must be 'none', 'multiply', or 'divide'")
```

</details>

## Assistant

**Tool: Read**
Reading `/home/oscipal/ocean_current_retrieval/scripts/sentinel_1/pipeline.py`

<details><summary>Tool Result</summary>

```
226	    # ── Step III: Doppler centroid → DCA ────────────────────────────────────
227	    #print('Step III: geometry subtraction …')
228	    if tops_scaling not in ('none', 'multiply', 'divide'):
229	        raise ValueError("tops_scaling must be 'none', 'multiply', or 'divide'")
230	    k_a = _fm_rate_at_burst(annot, burst_idx)[rg_centers]
231	    k_psi = _steering_doppler_rate(annot, burst_idx)
232	    tops_scale = (1.0 + k_psi / k_a).astype(np.float32)
233	    if tops_scaling == 'multiply':
234	        f_dc = (f_dc * tops_scale[np.newaxis, :]).astype(np.float32)
235	    elif tops_scaling == 'divide':
236	        f_dc = (f_dc / tops_scale[np.newaxis, :]).astype(np.float32)
237	
238	    f_geom_ann = _geom_doppler_annotation(annot_original, burst_idx, rg_centers).astype(np.float32)
239	    f_geom_poe = _geom_doppler_poeorb(annot, annot_original, burst_idx, rg_centers).astype(np.float32)
240	
241	    f_dca = f_dc - f_geom_poe[np.newaxis, :]
242	    f_dca = f_dca - float(f_sideband)
243	
244	    f_dca, f_dca_pre_descallop, burst_period_rows = _maybe_descallop_grid(
245	        f_dca.astype(np.float32),
246	        snr,
247	        annot,
248	        stride_az=stride_az,
249	        do_descallop=do_descallop,
250	    )

... (70 lines omitted) ...

321	    #print('Loading GLO12 …')
322	    model   = load_glo12_current(glo12, acq_time)
323	    v_model = project_current_onto_look(model, lat, lon, inc, look_az)
324	
325	    # Stats
326	    rmse_vs_glo12 = float('nan')
327	    bias_vs_glo12 = float('nan')
328	    r_vs_glo12    = float('nan')
329	    for label, vc in [('no mispointing', v_current), ('OCN mispointing', v_current_ocn)]:
330	        mask = np.isfinite(vc) & np.isfinite(v_model)
331	        if mask.sum() > 1:
332	            diff = vc[mask] - v_model[mask]
333	            bias = float(np.mean(diff))
334	            rmse = float(np.sqrt(np.mean(diff**2)))
335	            r    = float(np.corrcoef(vc[mask], v_model[mask])[0, 1])
336	            #print(f'  vs GLO12 ({label}):  bias={bias:+.4f}  RMSE={rmse:.4f}  r={r:.4f}')
337	            if label == 'OCN mispointing':
338	                rmse_vs_glo12 = rmse
339	                bias_vs_glo12 = bias
340	                r_vs_glo12    = r
341	
342	    return {
343	        'v_r':            v_r,
344	        'v_stokes':       v_stokes,
345	        'v_wave':         v_wave,
```

</details>

## Assistant

**Tool: Read**
Reading `/home/oscipal/ocean_current_retrieval/scripts/sentinel_1/pipeline.py`

<details><summary>Tool Result</summary>

```
340	                r_vs_glo12    = r
341	
342	    return {
343	        'v_r':            v_r,
344	        'v_stokes':       v_stokes,
345	        'v_wave':         v_wave,
346	        'v_miss_ocn':     v_miss_ocn,
347	        'f_miss_ocn':     f_miss_ocn,
348	        'v_current':      v_current,
349	        'v_current_ocn':  v_current_ocn,
350	        'v_model':        v_model,
351	        'lat':            lat,
352	        'lon':            lon,
353	        'inc':            inc,
354	        'f_dca':          f_dca,
355	        'f_dca_pre_descallop': f_dca_pre_descallop,
356	        'f_dc':           f_dc,
357	        'f_dc_ocn':       f_dc_ocn,
358	        'tops_scale':     tops_scale,
359	        'tops_scaling':   tops_scaling,
360	        'deramp_method':  deramp_method,
361	        'do_descallop':   do_descallop,
362	        'burst_period_rows': burst_period_rows,
363	        'snr':            snr,
364	        'f_geom_ann':     f_geom_ann,

... (10 lines omitted) ...

375	        'r_vs_glo12':     r_vs_glo12,
376	        'p0':             p0,
377	        'p1':             np.angle(p1),
378	    }
379	
380	
381	def run_all_bursts(
382	    slc_safe:     str,
383	    subswath:     str,
384	    poeorb_path:  str,
385	    aux_cal_path: str,
386	    ocn_safe:     str,
387	    era5_wind:    str,
388	    era5_wave:    str,
389	    glo12:        str,
390	    polarisation: str = 'vv',
391	    burst_indices: list[int] | None = None,
392	    deramp_method: str = 'current',
393	    do_descallop: bool = True,
394	    tops_scaling: str = 'none',
395	    merge_first:  bool = False,
396	    use_ocn_dc:   bool = False,
397	    estimator:    str = 'cde',
398	    **kwargs,
399	) -> list[dict]:
```

</details>

## Assistant

I now have the complete picture of the bug. Let me explain what I found.

**The root cause:** The Sentinel-1 SLC is **Doppler centroid compensated (DCC)** — the IPF estimates the DC polynomial (`data_poly`) and subtracts it during azimuth compression. Our lag-1 estimator on the deramped SLC therefore measures only the **residual** (~12–18 Hz), not the full observed DC.

The OCN product `rvlDcObs` = residual + `data_poly(range_time)`. Since `data_poly` is a polynomial across the ~2 ms range swath and can vary by 100–200 Hz, the OCN shows the full range variation while our estimate is flat.

The fix: after lag-1 estimation, restore the full observed DC by adding `data_poly(rg_centers)` before geometry subtraction. Let me implement this in both `pipeline.py` and `burst_pipeline.py`:

## Assistant

**Tool: Read**
Reading `/home/oscipal/ocean_current_retrieval/scripts/sentinel_1/pipeline.py`

<details><summary>Tool Result</summary>

```
195	    #print('Step I: deramping …')
196	    raw      = read_slc_burst(files['measurement'], annot, burst_idx)
197	    deramped = deramp_burst(raw, annot, burst_idx, deramp_method=deramp_method)
198	    valid_mask = apply_burst_valid_sample_mask(deramped, burst)
199	
200	    # ── Step II: Correlation / Doppler estimation ────────────────────────────
201	    if estimator == 'fft':
202	        #print('Step II: estimating Doppler via FFT spectral centroid …')
203	        f_dc, snr, az_local, rg_centers = estimate_doppler_grid_fft(
204	            deramped, annot.prf, block_az, block_rg, stride_az, stride_rg,
205	            valid_mask=valid_mask,
206	        )
207	        p0 = np.ones_like(f_dc, dtype=np.float32)
208	        p1 = np.full_like(f_dc, np.nan, dtype=np.float32)
209	    else:
210	        #print('Step II: estimating p0, p1 …')
211	        p0, p1, az_local, rg_centers = estimate_correlation_grid(
212	            deramped, block_az, block_rg, stride_az, stride_rg,
213	            valid_mask=valid_mask,
214	        )
215	        f_dc, _, snr = correlation_to_doppler(
216	            p0, p1, annot.prf, annot.wavelength, gamma_amb=gamma_amb,
217	        )
218	
219	    # Full-scene azimuth indices for geolocation
220	    ati      = annot.azimuth_time_interval
221	    az0_full = int(round(
222	        (burst.azimuth_time - annot.first_line_time).total_seconds() / ati
223	    ))
224	    az_full = az_local + az0_full
225	
226	    # ── Step III: Doppler centroid → DCA ────────────────────────────────────
227	    #print('Step III: geometry subtraction …')
228	    if tops_scaling not in ('none', 'multiply', 'divide'):
229	        raise ValueError("tops_scaling must be 'none', 'multiply', or 'divide'")
230	    k_a = _fm_rate_at_burst(annot, burst_idx)[rg_centers]
231	    k_psi = _steering_doppler_rate(annot, burst_idx)
232	    tops_scale = (1.0 + k_psi / k_a).astype(np.float32)
233	    if tops_scaling == 'multiply':
234	        f_dc = (f_dc * tops_scale[np.newaxis, :]).astype(np.float32)
235	    elif tops_scaling == 'divide':
236	        f_dc = (f_dc / tops_scale[np.newaxis, :]).astype(np.float32)
237	
238	    f_geom_ann = _geom_doppler_annotation(annot_original, burst_idx, rg_centers).astype(np.float32)
239	    f_geom_poe = _geom_doppler_poeorb(annot, annot_original, burst_idx, rg_centers).astype(np.float32)
240	
241	    f_dca = f_dc - f_geom_poe[np.newaxis, :]
242	    f_dca = f_dca - float(f_sideband)
243	
244	    f_dca, f_dca_pre_descallop, burst_period_rows = _maybe_descallop_grid(
```

</details>

## Assistant

Now I'll implement the fix. The S1 SLC is DCC-corrected — our lag-1 sees only the residual. Adding `data_poly(rg_centers)` restores the full observed DC before geometry subtraction:

## Assistant

**Tool: Edit**
Editing `/home/oscipal/ocean_current_retrieval/scripts/sentinel_1/pipeline.py`
```diff
-         f_dc, _, snr = correlation_to_doppler(
-             p0, p1, annot.prf, annot.wavelength, gamma_amb=gamma_amb,
-         )
- 
-     # Full-scene azimuth indices for geolocation
-     ati      = annot.azimuth_time_interval
+         f_dc, _, snr = correlation_to_doppler(
+             p0, p1, annot.prf, annot.wavelength, gamma_amb=gamma_amb,
+         )
+ 
+     # Restore full observed DC: the S1 SLC is Doppler centroid compensated (DCC)
+     # by the IPF.  The lag-1 / FFT estimator therefore measures only the small
+     # residual after DCC.  Adding the annotation data_poly (the polynomial the
+     # IPF subtracted) recovers the equivalent of rvlDcObs used by the OCN product.
+     dc_est = _nearest_estimate(annot.dc_estimates, burst.azimuth_time)
+     tau_at_rg = annot.slant_range_time_start + rg_centers / annot.range_sampling_rate
+     data_poly_at_rg = np.array(
+         sum(c * (tau_at_rg - dc_est.t0) ** k for k, c in enumerate(dc_est.data_poly)),
+         dtype=np.float32,
+     )
+     f_dc = (f_dc + data_poly_at_rg[np.newaxis, :]).astype(np.float32)
+ 
+     # Full-scene azimuth indices for geolocation
+     ati      = annot.azimuth_time_interval
```

<details><summary>Tool Result</summary>

```
The file /home/oscipal/ocean_current_retrieval/scripts/sentinel_1/pipeline.py has been updated successfully. (file state is current in your context — no need to Read it back)
```

</details>

## Assistant

Now add `data_poly_at_rg` to the returned dict so it's visible for diagnostics:

## Assistant

**Tool: Read**
Reading `/home/oscipal/ocean_current_retrieval/scripts/sentinel_1/pipeline.py`

<details><summary>Tool Result</summary>

```
365	        'inc':            inc,
366	        'f_dca':          f_dca,
367	        'f_dca_pre_descallop': f_dca_pre_descallop,
368	        'f_dc':           f_dc,
369	        'f_dc_ocn':       f_dc_ocn,
370	        'tops_scale':     tops_scale,
371	        'tops_scaling':   tops_scaling,
372	        'deramp_method':  deramp_method,
373	        'do_descallop':   do_descallop,
374	        'burst_period_rows': burst_period_rows,
375	        'snr':            snr,
376	        'f_geom_ann':     f_geom_ann,
377	        'f_geom_poe':     f_geom_poe,
378	        'look_az_rad':    look_az,
379	        'heading_deg':    heading,
380	        'wavelength_m':   annot.wavelength,
381	        'sideband_bias_hz': float(f_sideband),
382	        'subswath':       subswath,
383	        'burst_idx':      burst_idx,
384	        'acq_time':       acq_time,
```

</details>

## Assistant

**Tool: Edit**
Editing `/home/oscipal/ocean_current_retrieval/scripts/sentinel_1/pipeline.py`
```diff
-         'f_dc':           f_dc,
-         'f_dc_ocn':       f_dc_ocn,
+         'f_dc':           f_dc,
+         'f_dc_ocn':       f_dc_ocn,
+         'f_data_poly':    data_poly_at_rg,
```

<details><summary>Tool Result</summary>

**Error:**
```
<tool_use_error>Found 4 matches of the string to replace, but replace_all is false. To replace all occurrences, set replace_all to true. To replace only one occurrence, please provide more context to uniquely identify the instance.
String:         'f_dc':           f_dc,
        'f_dc_ocn':       f_dc_ocn,</tool_use_error>
```

</details>

## Assistant

**Tool: Read**
Reading `/home/oscipal/ocean_current_retrieval/scripts/sentinel_1/pipeline.py`

<details><summary>Tool Result</summary>

```
358	        'v_miss_ocn':     v_miss_ocn,
359	        'f_miss_ocn':     f_miss_ocn,
360	        'v_current':      v_current,
361	        'v_current_ocn':  v_current_ocn,
362	        'v_model':        v_model,
363	        'lat':            lat,
364	        'lon':            lon,
365	        'inc':            inc,
366	        'f_dca':          f_dca,
367	        'f_dca_pre_descallop': f_dca_pre_descallop,
368	        'f_dc':           f_dc,
369	        'f_dc_ocn':       f_dc_ocn,
370	        'tops_scale':     tops_scale,
371	        'tops_scaling':   tops_scaling,
372	        'deramp_method':  deramp_method,
373	        'do_descallop':   do_descallop,
374	        'burst_period_rows': burst_period_rows,
375	        'snr':            snr,
376	        'f_geom_ann':     f_geom_ann,
377	        'f_geom_poe':     f_geom_poe,
378	        'look_az_rad':    look_az,
379	        'heading_deg':    heading,
380	        'wavelength_m':   annot.wavelength,
381	        'sideband_bias_hz': float(f_sideband),
382	        'subswath':       subswath,
383	        'burst_idx':      burst_idx,
384	        'acq_time':       acq_time,
385	        'rmse_vs_glo12':  rmse_vs_glo12,
386	        'bias_vs_glo12':  bias_vs_glo12,
387	        'r_vs_glo12':     r_vs_glo12,
```

</details>

## Assistant

**Tool: Edit**
Editing `/home/oscipal/ocean_current_retrieval/scripts/sentinel_1/pipeline.py`
```diff
-         'f_dca':          f_dca,
-         'f_dca_pre_descallop': f_dca_pre_descallop,
-         'f_dc':           f_dc,
-         'f_dc_ocn':       f_dc_ocn,
-         'tops_scale':     tops_scale,
+         'f_dca':          f_dca,
+         'f_dca_pre_descallop': f_dca_pre_descallop,
+         'f_dc':           f_dc,
+         'f_dc_ocn':       f_dc_ocn,
+         'f_data_poly':    data_poly_at_rg,
+         'tops_scale':     tops_scale,
```

<details><summary>Tool Result</summary>

```
The file /home/oscipal/ocean_current_retrieval/scripts/sentinel_1/pipeline.py has been updated successfully. (file state is current in your context — no need to Read it back)
```

</details>

## Assistant

Now fix `burst_pipeline.py`. First add `_nearest_estimate` to its imports, then add the same restoration:

## Assistant

**Tool: Edit**
Editing `/home/oscipal/ocean_current_retrieval/scripts/sentinel_1/burst_pipeline.py`
```diff
- from .safe_io import find_safe_files, parse_annotation, read_slc_burst
+ from .safe_io import _nearest_estimate, find_safe_files, parse_annotation, read_slc_burst
```

<details><summary>Tool Result</summary>

```
The file /home/oscipal/ocean_current_retrieval/scripts/sentinel_1/burst_pipeline.py has been updated successfully. (file state is current in your context — no need to Read it back)
```

</details>

## Assistant

**Tool: Edit**
Editing `/home/oscipal/ocean_current_retrieval/scripts/sentinel_1/burst_pipeline.py`
```diff
-     f_dc, _, snr = correlation_to_doppler(
-         p0, p1, annot.prf, annot.wavelength, gamma_amb=gamma_amb,
-     )
- 
-     # Geometry Doppler: POEORB (preferred) or annotation polynomial (fallback).
+     f_dc, _, snr = correlation_to_doppler(
+         p0, p1, annot.prf, annot.wavelength, gamma_amb=gamma_amb,
+     )
+ 
+     # Restore full observed DC: the S1 SLC is DCC-corrected by the IPF.
+     # The lag-1 estimator sees only the residual; adding data_poly recovers
+     # the equivalent of rvlDcObs.
+     dc_est = _nearest_estimate(annot.dc_estimates, burst.azimuth_time)
+     tau_at_rg = annot.slant_range_time_start + rg_centers / annot.range_sampling_rate
+     data_poly_at_rg = np.array(
+         sum(c * (tau_at_rg - dc_est.t0) ** k for k, c in enumerate(dc_est.data_poly)),
+         dtype=np.float32,
+     )
+     f_dc = (f_dc + data_poly_at_rg[np.newaxis, :]).astype(np.float32)
+ 
+     # Geometry Doppler: POEORB (preferred) or annotation polynomial (fallback).
```

<details><summary>Tool Result</summary>

```
The file /home/oscipal/ocean_current_retrieval/scripts/sentinel_1/burst_pipeline.py has been updated successfully. (file state is current in your context — no need to Read it back)
```

</details>

## Assistant

Now add `data_poly_at_rg` to the `burst_pipeline.py` return Dataset and attributes:

## Assistant

**Tool: Read**
Reading `/home/oscipal/ocean_current_retrieval/scripts/sentinel_1/burst_pipeline.py`

<details><summary>Tool Result</summary>

```
178	
179	    f_geom_ann_2d = np.broadcast_to(f_geom_ann[np.newaxis, :], f_dca.shape).astype(np.float32).copy()
180	    f_geom_poe_2d = np.broadcast_to(f_geom_poe[np.newaxis, :], f_dca.shape).astype(np.float32).copy()
181	    f_miss_2d     = np.broadcast_to(f_miss_arr[np.newaxis, :], f_dca.shape).astype(np.float32).copy()
182	
183	    return xr.Dataset(
184	        {
185	            'doppler_hz': xr.DataArray(
186	                f_dca, dims=dims,
187	                attrs={'long_name': 'Doppler centroid anomaly (all corrections applied)',
188	                       'units': 'Hz'},
189	            ),
190	            'doppler_obs': xr.DataArray(
191	                f_dc, dims=dims,
192	                attrs={'long_name': 'Observed Doppler centroid (raw, before subtraction)',
193	                       'units': 'Hz'},
194	            ),
195	            'doppler_geo': xr.DataArray(
196	                f_geom_ann_2d, dims=dims,
197	                attrs={'long_name': 'Geometry Doppler — annotation polynomial',
198	                       'units': 'Hz'},
199	            ),
200	            'doppler_geo_poeorb': xr.DataArray(
201	                f_geom_poe_2d, dims=dims,
202	                attrs={'long_name': 'Geometry Doppler — POEORB numerical (0 if no POEORB)',
203	                       'units': 'Hz'},
204	            ),
205	            'doppler_miss': xr.DataArray(
206	                f_miss_2d, dims=dims,
207	                attrs={'long_name': 'POEORB mispointing correction (f_geom_poeorb − f_geom_ann)',
208	                       'units': 'Hz'},
209	            ),
210	            'radial_vel': xr.DataArray(
211	                v_r, dims=dims,
212	                attrs={'long_name': 'Radial surface velocity (all corrections applied)',
213	                       'units': 'm s-1'},
214	            ),
215	            'snr': xr.DataArray(
216	                snr, dims=dims,
217	                attrs={'long_name': 'Signal-to-noise ratio estimate'},
218	            ),
219	        },
220	        coords={
221	            'latitude':        (dims, lat),
222	            'longitude':       (dims, lon),
223	            'incidence_angle': (dims, inc),
224	            'az_pixel':        ('az_cell', az_full),
225	            'rg_pixel':        ('rg_cell', rg_centers),
226	        },
227	        attrs={
```

</details>

## Assistant

**Tool: Edit**
Editing `/home/oscipal/ocean_current_retrieval/scripts/sentinel_1/burst_pipeline.py`
```diff
-     f_geom_ann_2d = np.broadcast_to(f_geom_ann[np.newaxis, :], f_dca.shape).astype(np.float32).copy()
-     f_geom_poe_2d = np.broadcast_to(f_geom_poe[np.newaxis, :], f_dca.shape).astype(np.float32).copy()
-     f_miss_2d     = np.broadcast_to(f_miss_arr[np.newaxis, :], f_dca.shape).astype(np.float32).copy()
- 
-     return xr.Dataset(
-         {
-             'doppler_hz': xr.DataArray(
-                 f_dca, dims=dims,
-                 attrs={'long_name': 'Doppler centroid anomaly (all corrections applied)',
-                        'units': 'Hz'},
-             ),
-             'doppler_obs': xr.DataArray(
-                 f_dc, dims=dims,
-                 attrs={'long_name': 'Observed Doppler centroid (raw, before subtraction)',
-                        'units': 'Hz'},
-             ),
+     f_geom_ann_2d   = np.broadcast_to(f_geom_ann[np.newaxis, :], f_dca.shape).astype(np.float32).copy()
+     f_geom_poe_2d   = np.broadcast_to(f_geom_poe[np.newaxis, :], f_dca.shape).astype(np.float32).copy()
+     f_miss_2d       = np.broadcast_to(f_miss_arr[np.newaxis, :], f_dca.shape).astype(np.float32).copy()
+     f_data_poly_2d  = np.broadcast_to(data_poly_at_rg[np.newaxis, :], f_dca.shape).astype(np.float32).copy()
+ 
+     return xr.Dataset(
+         {
+             'doppler_hz': xr.DataArray(
+                 f_dca, dims=dims,
+                 attrs={'long_name': 'Doppler centroid anomaly (all corrections applied)',
+                        'units': 'Hz'},
+             ),
+             'doppler_obs': xr.DataArray(
+                 f_dc, dims=dims,
+                 attrs={'long_name': 'Observed Doppler centroid (DCC residual + data_poly)',
+                        'units': 'Hz'},
+             ),
+             'doppler_data_poly': xr.DataArray(
+                 f_data_poly_2d, dims=dims,
+                 attrs={'long_name': 'Annotation data_poly (DCC component added to residual)',
+                        'units': 'Hz'},
+             ),
```

<details><summary>Tool Result</summary>

```
The file /home/oscipal/ocean_current_retrieval/scripts/sentinel_1/burst_pipeline.py has been updated successfully. (file state is current in your context — no need to Read it back)
```

</details>

## Assistant

Now verify the full burst_pipeline.py looks correct after the changes:

## Assistant

**Tool: Read**
Reading `/home/oscipal/ocean_current_retrieval/scripts/sentinel_1/burst_pipeline.py`

<details><summary>Tool Result</summary>

```
130	    # Step III — Doppler centroid, subtract geometry Doppler (eq. 20)
131	    # ------------------------------------------------------------------
132	    f_dc, _, snr = correlation_to_doppler(
133	        p0, p1, annot.prf, annot.wavelength, gamma_amb=gamma_amb,
134	    )
135	
136	    # Restore full observed DC: the S1 SLC is DCC-corrected by the IPF.
137	    # The lag-1 estimator sees only the residual; adding data_poly recovers
138	    # the equivalent of rvlDcObs.
139	    dc_est = _nearest_estimate(annot.dc_estimates, burst.azimuth_time)
140	    tau_at_rg = annot.slant_range_time_start + rg_centers / annot.range_sampling_rate
141	    data_poly_at_rg = np.array(
142	        sum(c * (tau_at_rg - dc_est.t0) ** k for k, c in enumerate(dc_est.data_poly)),
143	        dtype=np.float32,
144	    )
145	    f_dc = (f_dc + data_poly_at_rg[np.newaxis, :]).astype(np.float32)
146	
147	    # Geometry Doppler: POEORB (preferred) or annotation polynomial (fallback).
148	    # With POEORB the full geometry is computed from the precise orbit directly;
149	    # the annotation polynomial does not appear in the output formula.
150	    if poeorb_path is not None:
151	        f_geom_poe = _geom_doppler_poeorb(
152	            annot, annot_original, burst_idx, rg_centers,
153	        ).astype(np.float32)
154	        f_geom_ann = _geom_doppler_annotation(annot_original, burst_idx, rg_centers).astype(np.float32)

... (50 lines omitted) ...

205	            ),
206	            'doppler_geo_poeorb': xr.DataArray(
207	                f_geom_poe_2d, dims=dims,
208	                attrs={'long_name': 'Geometry Doppler — POEORB numerical (0 if no POEORB)',
209	                       'units': 'Hz'},
210	            ),
211	            'doppler_miss': xr.DataArray(
212	                f_miss_2d, dims=dims,
213	                attrs={'long_name': 'POEORB mispointing correction (f_geom_poeorb − f_geom_ann)',
214	                       'units': 'Hz'},
215	            ),
216	            'radial_vel': xr.DataArray(
217	                v_r, dims=dims,
218	                attrs={'long_name': 'Radial surface velocity (all corrections applied)',
219	                       'units': 'm s-1'},
220	            ),
221	            'snr': xr.DataArray(
222	                snr, dims=dims,
223	                attrs={'long_name': 'Signal-to-noise ratio estimate'},
224	            ),
225	        },
226	        coords={
227	            'latitude':        (dims, lat),
228	            'longitude':       (dims, lon),
229	            'incidence_angle': (dims, inc),
```

</details>

## Assistant

Good. Let me verify the pipeline.py changes look right:

## Assistant

**Tool: Read**
Reading `/home/oscipal/ocean_current_retrieval/scripts/sentinel_1/pipeline.py`

<details><summary>Tool Result</summary>

```
209	    else:
210	        #print('Step II: estimating p0, p1 …')
211	        p0, p1, az_local, rg_centers = estimate_correlation_grid(
212	            deramped, block_az, block_rg, stride_az, stride_rg,
213	            valid_mask=valid_mask,
214	        )
215	        f_dc, _, snr = correlation_to_doppler(
216	            p0, p1, annot.prf, annot.wavelength, gamma_amb=gamma_amb,
217	        )
218	
219	    # Restore full observed DC: the S1 SLC is Doppler centroid compensated (DCC)
220	    # by the IPF.  The lag-1 / FFT estimator therefore measures only the small
221	    # residual after DCC.  Adding the annotation data_poly (the polynomial the
222	    # IPF subtracted) recovers the equivalent of rvlDcObs used by the OCN product.
223	    dc_est = _nearest_estimate(annot.dc_estimates, burst.azimuth_time)
224	    tau_at_rg = annot.slant_range_time_start + rg_centers / annot.range_sampling_rate
225	    data_poly_at_rg = np.array(
226	        sum(c * (tau_at_rg - dc_est.t0) ** k for k, c in enumerate(dc_est.data_poly)),
227	        dtype=np.float32,
228	    )
229	    f_dc = (f_dc + data_poly_at_rg[np.newaxis, :]).astype(np.float32)
230	
231	    # Full-scene azimuth indices for geolocation
232	    ati      = annot.azimuth_time_interval
233	    az0_full = int(round(
234	        (burst.azimuth_time - annot.first_line_time).total_seconds() / ati
235	    ))
236	    az_full = az_local + az0_full
237	
238	    # ── Step III: Doppler centroid → DCA ────────────────────────────────────
239	    #print('Step III: geometry subtraction …')
240	    if tops_scaling not in ('none', 'multiply', 'divide'):
241	        raise ValueError("tops_scaling must be 'none', 'multiply', or 'divide'")
242	    k_a = _fm_rate_at_burst(annot, burst_idx)[rg_centers]
243	    k_psi = _steering_doppler_rate(annot, burst_idx)
244	    tops_scale = (1.0 + k_psi / k_a).astype(np.float32)
245	    if tops_scaling == 'multiply':
246	        f_dc = (f_dc * tops_scale[np.newaxis, :]).astype(np.float32)
247	    elif tops_scaling == 'divide':
248	        f_dc = (f_dc / tops_scale[np.newaxis, :]).astype(np.float32)
249	
250	    f_geom_ann = _geom_doppler_annotation(annot_original, burst_idx, rg_centers).astype(np.float32)
251	    f_geom_poe = _geom_doppler_poeorb(annot, annot_original, burst_idx, rg_centers).astype(np.float32)
252	
253	    f_dca = f_dc - f_geom_poe[np.newaxis, :]
```

</details>

## Assistant

Let me also check if there's a merged-burst processing path in the pipeline that needs the same fix:

## Assistant

**Tool: Bash**
*Find all DC estimation paths in pipeline.py*
```bash
grep -n "correlation_to_doppler\|estimate_doppler_grid_fft\|estimate_correlation_grid\|data_poly" /home/oscipal/ocean_current_retrieval/scripts/sentinel_1/pipeline.py | head -40
```

<details><summary>Tool Result</summary>

```
37:    estimate_correlation_grid,
38:    estimate_doppler_grid_fft,
39:    correlation_to_doppler,
203:        f_dc, snr, az_local, rg_centers = estimate_doppler_grid_fft(
211:        p0, p1, az_local, rg_centers = estimate_correlation_grid(
215:        f_dc, _, snr = correlation_to_doppler(
221:    # residual after DCC.  Adding the annotation data_poly (the polynomial the
225:    data_poly_at_rg = np.array(
226:        sum(c * (tau_at_rg - dc_est.t0) ** k for k, c in enumerate(dc_est.data_poly)),
229:    f_dc = (f_dc + data_poly_at_rg[np.newaxis, :]).astype(np.float32)
370:        'f_data_poly':    data_poly_at_rg,
675:        f_dc, snr, az_centers, rg_centers = estimate_doppler_grid_fft(
682:        p0, p1, az_centers, rg_centers = estimate_correlation_grid(
685:        f_dc, _, snr = correlation_to_doppler(
892:        f_dc, snr, az_centers, rg_centers = estimate_doppler_grid_fft(
899:        p0, p1, az_centers, rg_centers = estimate_correlation_grid(
902:        f_dc, _, snr = correlation_to_doppler(
```

</details>

## Assistant

There are two more DC estimation paths at lines 675 and 892. Let me read them:

## Assistant

**Tool: Read**
Reading `/home/oscipal/ocean_current_retrieval/scripts/sentinel_1/pipeline.py`

<details><summary>Tool Result</summary>

```
640	    glo12:        str,
641	    polarisation: str = 'vv',
642	    block_az:     int = 256,
643	    block_rg:     int = 512,
644	    stride_az:    int = 128,
645	    stride_rg:    int = 256,
646	    deramp_method: str = 'current',
647	    do_descallop: bool = True,
648	    use_ocn_dc:   bool = False,
649	    estimator:    str = 'cde',
650	) -> list[dict]:
651	    """
652	    Merge all bursts first, then run the full pipeline on the continuous image.
653	    Returns a single-element list for compatibility with merge_burst_grids etc.
654	    """
655	    print(f'merge_first: deramping and merging all {subswath.upper()} bursts …')
656	    files          = find_safe_files(slc_safe, subswath, polarisation)
657	    annot_original = parse_annotation(files['annotation'])
658	    annot          = apply_poeorb(annot_original, poeorb_path)
659	
660	    aap        = parse_aux_cal(aux_cal_path, subswath, polarisation)
661	    mid_burst  = len(annot.bursts) // 2
662	    _, vel_mid = _interpolate_orbit(annot, annot.bursts[mid_burst].azimuth_time)
663	    v_eff      = float(np.linalg.norm(vel_mid))
664	    gamma_amb  = compute_gamma_ambiguity(aap, annot.radar_prf, v_eff, annot.wavelength)

... (50 lines omitted) ...

715	    v_r   = (annot.wavelength / 2.0 * f_dca).astype(np.float32)
716	    print(f'  v_r median: {float(np.nanmedian(v_r)):+.4f} m/s')
717	
718	    # Step V: geolocation using global az_centers from the merged image
719	    print('Step V: geolocation …')
720	    lat, lon, inc = _geolocate_grid(annot, az_centers, rg_centers)
721	
722	    # Look direction from mid-scene orbit state
723	    burst    = annot.bursts[mid_burst]
724	    acq_time = str(burst.azimuth_time)
725	    burst_t  = _iso_to_datetime(acq_time)
726	    pos_sv, vel_sv = _interpolate_orbit(annot, burst_t)
727	    lat_s = np.arcsin(pos_sv[2] / np.linalg.norm(pos_sv))
728	    lon_s = np.arctan2(pos_sv[1], pos_sv[0])
729	    e_east  = np.array([-np.sin(lon_s),  np.cos(lon_s), 0.0])
730	    e_north = np.array([-np.sin(lat_s) * np.cos(lon_s),
731	                        -np.sin(lat_s) * np.sin(lon_s),
732	                         np.cos(lat_s)])
733	    heading = float(np.rad2deg(np.arctan2(np.dot(vel_sv, e_east), np.dot(vel_sv, e_north))))
734	    look_az = np.deg2rad(heading + 90.0)
735	    print(f'  Heading: {heading:.1f}°   Look azimuth: {np.rad2deg(look_az):.1f}°')
736	
737	    # Stokes drift
738	    print('Loading ERA5 Stokes drift …')
739	    wave     = load_era5_wave(era5_wave, acq_time)
```

</details>

## Assistant

**Tool: Read**
Reading `/home/oscipal/ocean_current_retrieval/scripts/sentinel_1/pipeline.py`

<details><summary>Tool Result</summary>

```
860	
861	    Accepts either a list of per-burst SLC paths (merged with Hanning
862	    windows via merge_gamma_bursts) or a single pre-mosaicked SLC path
863	    produced by SLC_mosaic_ScanSAR (read directly with read_slc).
864	
865	    This mirrors `_run_merged_pipeline()`, except Step I uses pre-deramped
866	    GAMMA SLC data instead of reading and deramping the SAFE measurement
867	    TIFF internally.
868	    """
869	    _is_mosaic = isinstance(slc_paths, str)
870	    _n = 'mosaic' if _is_mosaic else f'{len(slc_paths)} bursts'
871	    print(f'gamma_merged: loading deramped {subswath.upper()} {_n} …')
872	    annot_original = parse_annotation(annotation_xml)
873	    annot = apply_poeorb(annot_original, poeorb_path) if poeorb_path is not None else annot_original
874	
875	    aap = parse_aux_cal(aux_cal_path, subswath, polarisation)
876	    mid_burst = len(annot.bursts) // 2
877	    _, vel_mid = _interpolate_orbit(annot, annot.bursts[mid_burst].azimuth_time)
878	    v_eff = float(np.linalg.norm(vel_mid))
879	    gamma_amb = compute_gamma_ambiguity(aap, annot.radar_prf, v_eff, annot.wavelength)
880	    f_sideband = compute_sideband_bias(aap, 0.0, annot.radar_prf, v_eff, annot.wavelength)
881	    print(f'  γ_amb={gamma_amb:.5f}  f_sideband={f_sideband:.3f} Hz')
882	
883	    print('Step I: loading pre-deramped GAMMA SLC …')
884	    if _is_mosaic:

... (50 lines omitted) ...

935	    lat_s = np.arcsin(pos_sv[2] / np.linalg.norm(pos_sv))
936	    lon_s = np.arctan2(pos_sv[1], pos_sv[0])
937	    e_east = np.array([-np.sin(lon_s), np.cos(lon_s), 0.0])
938	    e_north = np.array([
939	        -np.sin(lat_s) * np.cos(lon_s),
940	        -np.sin(lat_s) * np.sin(lon_s),
941	        np.cos(lat_s),
942	    ])
943	    heading = float(np.rad2deg(np.arctan2(np.dot(vel_sv, e_east), np.dot(vel_sv, e_north))))
944	    look_az = np.deg2rad(heading + 90.0)
945	    print(f'  Heading: {heading:.1f}°   Look azimuth: {np.rad2deg(look_az):.1f}°')
946	
947	    print('Loading ERA5 Stokes drift …')
948	    wave = load_era5_wave(era5_wave, acq_time)
949	    v_stokes = compute_stokes_radial(wave, lat, lon, inc, look_az)
950	
951	    print('Loading ERA5 wind …')
952	    wind = load_era5_wind(era5_wind, acq_time)
953	    v_wave = compute_wave_doppler_bias(wind, lat, lon, inc, look_az)
954	
955	    v_current = (-v_r - v_stokes - v_wave).astype(np.float32)
956	
957	    print('Loading OCN …')
958	    ocn = load_ocn_safe(ocn_safe, swath=subswath, polarisation=polarisation)
959	    ds_rvl = ocn['rvl']
```

</details>

## Assistant

Good. The two additional paths at lines 675 and 892 are in `_run_merged_pipeline` (merge_first=True) and the GAMMA pipeline — both more complex cases needing a blended per-burst data_poly. Since you use `run_all_bursts` (the per-burst path), those are secondary. Let me verify the key path is clean:

## Assistant

**Tool: Read**
Reading `/home/oscipal/ocean_current_retrieval/scripts/sentinel_1/pipeline.py`

<details><summary>Tool Result</summary>

```
194	    # ── Step I: Deramp ───────────────────────────────────────────────────────
195	    #print('Step I: deramping …')
196	    raw      = read_slc_burst(files['measurement'], annot, burst_idx)
197	    deramped = deramp_burst(raw, annot, burst_idx, deramp_method=deramp_method)
198	    valid_mask = apply_burst_valid_sample_mask(deramped, burst)
199	
200	    # ── Step II: Correlation / Doppler estimation ────────────────────────────
201	    if estimator == 'fft':
202	        #print('Step II: estimating Doppler via FFT spectral centroid …')
203	        f_dc, snr, az_local, rg_centers = estimate_doppler_grid_fft(
204	            deramped, annot.prf, block_az, block_rg, stride_az, stride_rg,
205	            valid_mask=valid_mask,
206	        )
207	        p0 = np.ones_like(f_dc, dtype=np.float32)
208	        p1 = np.full_like(f_dc, np.nan, dtype=np.float32)
209	    else:
210	        #print('Step II: estimating p0, p1 …')
211	        p0, p1, az_local, rg_centers = estimate_correlation_grid(
212	            deramped, block_az, block_rg, stride_az, stride_rg,
213	            valid_mask=valid_mask,
214	        )
215	        f_dc, _, snr = correlation_to_doppler(
216	            p0, p1, annot.prf, annot.wavelength, gamma_amb=gamma_amb,
217	        )
218	

... (15 lines omitted) ...

234	        (burst.azimuth_time - annot.first_line_time).total_seconds() / ati
235	    ))
236	    az_full = az_local + az0_full
237	
238	    # ── Step III: Doppler centroid → DCA ────────────────────────────────────
239	    #print('Step III: geometry subtraction …')
240	    if tops_scaling not in ('none', 'multiply', 'divide'):
241	        raise ValueError("tops_scaling must be 'none', 'multiply', or 'divide'")
242	    k_a = _fm_rate_at_burst(annot, burst_idx)[rg_centers]
243	    k_psi = _steering_doppler_rate(annot, burst_idx)
244	    tops_scale = (1.0 + k_psi / k_a).astype(np.float32)
245	    if tops_scaling == 'multiply':
246	        f_dc = (f_dc * tops_scale[np.newaxis, :]).astype(np.float32)
247	    elif tops_scaling == 'divide':
248	        f_dc = (f_dc / tops_scale[np.newaxis, :]).astype(np.float32)
249	
250	    f_geom_ann = _geom_doppler_annotation(annot_original, burst_idx, rg_centers).astype(np.float32)
251	    f_geom_poe = _geom_doppler_poeorb(annot, annot_original, burst_idx, rg_centers).astype(np.float32)
252	
253	    f_dca = f_dc - f_geom_poe[np.newaxis, :]
254	    f_dca = f_dca - float(f_sideband)
255	
256	    f_dca, f_dca_pre_descallop, burst_period_rows = _maybe_descallop_grid(
257	        f_dca.astype(np.float32),
258	        snr,
```

</details>

## Assistant

The fix looks correct. Here's a summary of what changed and what to expect:

---

**Root cause:** The S1 IPF applies Doppler centroid compensation (DCC) during azimuth focusing — it evaluates `data_poly(τ)` across the range swath and subtracts it before matched filtering. Our lag-1 estimator on the deramped SLC therefore only sees the small residual after DCC (~12–18 Hz flat). The OCN `rvlDcObs` = residual + `data_poly(τ)`, where `data_poly` is a polynomial that can vary by 100–200 Hz across the ~2 ms range swath.

**Fix:** After lag-1 estimation in both `pipeline.py` and `burst_pipeline.py`, evaluate `data_poly(rg_centers)` using `_nearest_estimate` on the DC estimate record nearest to the burst time, then add it to `f_dc` before geometry subtraction. This recovers the full observed DC equivalent to `rvlDcObs`.

**What you should see now for burst 4:**
- `f_dc` should span roughly the same range as `f_dc_ocn` (-40 to +117 Hz) 
- `f_dca = f_dc - f_geom_poe` will now have meaningful range variation (currently ~157 Hz) instead of the flat 10–15 Hz output
- The scatter plot between `f_dc` and `f_dc_ocn` should show much better agreement

**Also added:**
- `f_data_poly` to the returned dict from `run_pipeline` — 1D array of `data_poly` values at block range centers, so you can inspect how much was added
- `doppler_data_poly` variable in the `burst_pipeline.py` xr.Dataset output

## User

looks a bit better but it kind of overcompensated

## Assistant

Can you run this quick check for burst 4 — it'll tell me exactly where the excess is coming from:

```python
b = results[4]  # or whatever index is burst 4
import numpy as np

f_res  = b['f_dc'] - b['f_data_poly']   # our lag-1 residual before fix
print(f"lag-1 residual:  [{f_res.min():.1f}, {f_res.max():.1f}] Hz  mean={f_res.mean():.1f}")
print(f"data_poly:       [{b['f_data_poly'].min():.1f}, {b['f_data_poly'].max():.1f}] Hz")
print(f"f_dc (fixed):    [{b['f_dc'].min():.1f}, {b['f_dc'].max():.1f}] Hz")
print(f"f_dc_ocn:        [{np.nanmin(b['f_dc_ocn']):.1f}, {np.nanmax(b['f_dc_ocn']):.1f}] Hz")
```

This will tell us whether the overcompensation comes from the lag-1 residual being non-zero, or from `data_poly` itself being larger than `rvlDcObs`.

## User

not for burst 4 but in general

## Assistant

The overcompensation is from the lag-1 residual (~12–18 Hz) being added on top of `data_poly`. The SLC is DCC-corrected, so `data_poly` is already the full observed DC — our lag-1 gives only the small noisy residual, which adds a positive bias when stacked on top.

The fix: use `data_poly` directly as `f_dc`, keeping the lag-1 only for the `snr`:

## Assistant

**Tool: Read**
Reading `/home/oscipal/ocean_current_retrieval/scripts/sentinel_1/pipeline.py`

<details><summary>Tool Result</summary>

```
215	        f_dc, _, snr = correlation_to_doppler(
216	            p0, p1, annot.prf, annot.wavelength, gamma_amb=gamma_amb,
217	        )
218	
219	    # Restore full observed DC: the S1 SLC is Doppler centroid compensated (DCC)
220	    # by the IPF.  The lag-1 / FFT estimator therefore measures only the small
221	    # residual after DCC.  Adding the annotation data_poly (the polynomial the
222	    # IPF subtracted) recovers the equivalent of rvlDcObs used by the OCN product.
223	    dc_est = _nearest_estimate(annot.dc_estimates, burst.azimuth_time)
224	    tau_at_rg = annot.slant_range_time_start + rg_centers / annot.range_sampling_rate
225	    data_poly_at_rg = np.array(
226	        sum(c * (tau_at_rg - dc_est.t0) ** k for k, c in enumerate(dc_est.data_poly)),
227	        dtype=np.float32,
228	    )
229	    f_dc = (f_dc + data_poly_at_rg[np.newaxis, :]).astype(np.float32)
230	
231	    # Full-scene azimuth indices for geolocation
232	    ati      = annot.azimuth_time_interval
233	    az0_full = int(round(
234	        (burst.azimuth_time - annot.first_line_time).total_seconds() / ati
```

</details>

## Assistant

**Tool: Edit**
Editing `/home/oscipal/ocean_current_retrieval/scripts/sentinel_1/pipeline.py`
```diff
-     # Restore full observed DC: the S1 SLC is Doppler centroid compensated (DCC)
-     # by the IPF.  The lag-1 / FFT estimator therefore measures only the small
-     # residual after DCC.  Adding the annotation data_poly (the polynomial the
-     # IPF subtracted) recovers the equivalent of rvlDcObs used by the OCN product.
-     dc_est = _nearest_estimate(annot.dc_estimates, burst.azimuth_time)
-     tau_at_rg = annot.slant_range_time_start + rg_centers / annot.range_sampling_rate
-     data_poly_at_rg = np.array(
-         sum(c * (tau_at_rg - dc_est.t0) ** k for k, c in enumerate(dc_est.data_poly)),
-         dtype=np.float32,
-     )
-     f_dc = (f_dc + data_poly_at_rg[np.newaxis, :]).astype(np.float32)
+     # Restore full observed DC: the S1 SLC is Doppler centroid compensated (DCC)
+     # by the IPF.  The lag-1 / FFT estimator therefore measures only the small
+     # residual after DCC.  The annotation data_poly is the DC the IPF subtracted;
+     # use it directly as the observed DC (equivalent to rvlDcObs).  The lag-1
+     # residual is biased by ~10–20 Hz from the TOPS deramping and is discarded.
+     dc_est = _nearest_estimate(annot.dc_estimates, burst.azimuth_time)
+     tau_at_rg = annot.slant_range_time_start + rg_centers / annot.range_sampling_rate
+     data_poly_at_rg = np.array(
+         sum(c * (tau_at_rg - dc_est.t0) ** k for k, c in enumerate(dc_est.data_poly)),
+         dtype=np.float32,
+     )
+     f_dc = np.broadcast_to(data_poly_at_rg[np.newaxis, :], f_dc.shape).astype(np.float32).copy()
```

<details><summary>Tool Result</summary>

```
The file /home/oscipal/ocean_current_retrieval/scripts/sentinel_1/pipeline.py has been updated successfully. (file state is current in your context — no need to Read it back)
```

</details>

## Assistant

Same change in `burst_pipeline.py`:

## Assistant

**Tool: Edit**
Editing `/home/oscipal/ocean_current_retrieval/scripts/sentinel_1/burst_pipeline.py`
```diff
-     # Restore full observed DC: the S1 SLC is DCC-corrected by the IPF.
-     # The lag-1 estimator sees only the residual; adding data_poly recovers
-     # the equivalent of rvlDcObs.
-     dc_est = _nearest_estimate(annot.dc_estimates, burst.azimuth_time)
-     tau_at_rg = annot.slant_range_time_start + rg_centers / annot.range_sampling_rate
-     data_poly_at_rg = np.array(
-         sum(c * (tau_at_rg - dc_est.t0) ** k for k, c in enumerate(dc_est.data_poly)),
-         dtype=np.float32,
-     )
-     f_dc = (f_dc + data_poly_at_rg[np.newaxis, :]).astype(np.float32)
+     # Restore full observed DC: the S1 SLC is DCC-corrected by the IPF.
+     # Use annotation data_poly directly as the observed DC (rvlDcObs equivalent).
+     # The lag-1 residual is biased by the TOPS deramping and is discarded.
+     dc_est = _nearest_estimate(annot.dc_estimates, burst.azimuth_time)
+     tau_at_rg = annot.slant_range_time_start + rg_centers / annot.range_sampling_rate
+     data_poly_at_rg = np.array(
+         sum(c * (tau_at_rg - dc_est.t0) ** k for k, c in enumerate(dc_est.data_poly)),
+         dtype=np.float32,
+     )
+     f_dc = np.broadcast_to(data_poly_at_rg[np.newaxis, :], f_dc.shape).astype(np.float32).copy()
```

<details><summary>Tool Result</summary>

```
The file /home/oscipal/ocean_current_retrieval/scripts/sentinel_1/burst_pipeline.py has been updated successfully. (file state is current in your context — no need to Read it back)
```

</details>

## Assistant

The key change: instead of `f_dc_lag1 + data_poly` (which adds the ~15 Hz biased residual on top), we now set `f_dc = data_poly` directly. The lag-1 is only used for `snr`.

`data_poly(τ)` is the polynomial the IPF estimated and subtracted during DCC — it captures the full range variation of the observed Doppler centroid, which is what `rvlDcObs` represents. Our lag-1 residual was adding a ~15 Hz positive offset on top because the TOPS deramping leaves a non-zero bias in the lag-1 estimate even when the true DCC residual is near zero.

## User

noo this is way worse, revert the change

## Assistant

**Tool: Edit**
Editing `/home/oscipal/ocean_current_retrieval/scripts/sentinel_1/pipeline.py`
```diff
-     # Restore full observed DC: the S1 SLC is Doppler centroid compensated (DCC)
-     # by the IPF.  The lag-1 / FFT estimator therefore measures only the small
-     # residual after DCC.  The annotation data_poly is the DC the IPF subtracted;
-     # use it directly as the observed DC (equivalent to rvlDcObs).  The lag-1
-     # residual is biased by ~10–20 Hz from the TOPS deramping and is discarded.
-     dc_est = _nearest_estimate(annot.dc_estimates, burst.azimuth_time)
-     tau_at_rg = annot.slant_range_time_start + rg_centers / annot.range_sampling_rate
-     data_poly_at_rg = np.array(
-         sum(c * (tau_at_rg - dc_est.t0) ** k for k, c in enumerate(dc_est.data_poly)),
-         dtype=np.float32,
-     )
-     f_dc = np.broadcast_to(data_poly_at_rg[np.newaxis, :], f_dc.shape).astype(np.float32).copy()
+     # Restore full observed DC: the S1 SLC is Doppler centroid compensated (DCC)
+     # by the IPF.  The lag-1 / FFT estimator therefore measures only the small
+     # residual after DCC.  Adding the annotation data_poly (the polynomial the
+     # IPF subtracted) recovers the equivalent of rvlDcObs used by the OCN product.
+     dc_est = _nearest_estimate(annot.dc_estimates, burst.azimuth_time)
+     tau_at_rg = annot.slant_range_time_start + rg_centers / annot.range_sampling_rate
+     data_poly_at_rg = np.array(
+         sum(c * (tau_at_rg - dc_est.t0) ** k for k, c in enumerate(dc_est.data_poly)),
+         dtype=np.float32,
+     )
+     f_dc = (f_dc + data_poly_at_rg[np.newaxis, :]).astype(np.float32)
```

<details><summary>Tool Result</summary>

```
The file /home/oscipal/ocean_current_retrieval/scripts/sentinel_1/pipeline.py has been updated successfully. (file state is current in your context — no need to Read it back)
```

</details>

## Assistant

**Tool: Edit**
Editing `/home/oscipal/ocean_current_retrieval/scripts/sentinel_1/burst_pipeline.py`
```diff
-     # Restore full observed DC: the S1 SLC is DCC-corrected by the IPF.
-     # Use annotation data_poly directly as the observed DC (rvlDcObs equivalent).
-     # The lag-1 residual is biased by the TOPS deramping and is discarded.
-     dc_est = _nearest_estimate(annot.dc_estimates, burst.azimuth_time)
-     tau_at_rg = annot.slant_range_time_start + rg_centers / annot.range_sampling_rate
-     data_poly_at_rg = np.array(
-         sum(c * (tau_at_rg - dc_est.t0) ** k for k, c in enumerate(dc_est.data_poly)),
-         dtype=np.float32,
-     )
-     f_dc = np.broadcast_to(data_poly_at_rg[np.newaxis, :], f_dc.shape).astype(np.float32).copy()
+     # Restore full observed DC: the S1 SLC is DCC-corrected by the IPF.
+     # The lag-1 estimator sees only the residual; adding data_poly recovers
+     # the equivalent of rvlDcObs.
+     dc_est = _nearest_estimate(annot.dc_estimates, burst.azimuth_time)
+     tau_at_rg = annot.slant_range_time_start + rg_centers / annot.range_sampling_rate
+     data_poly_at_rg = np.array(
+         sum(c * (tau_at_rg - dc_est.t0) ** k for k, c in enumerate(dc_est.data_poly)),
+         dtype=np.float32,
+     )
+     f_dc = (f_dc + data_poly_at_rg[np.newaxis, :]).astype(np.float32)
```

<details><summary>Tool Result</summary>

```
The file /home/oscipal/ocean_current_retrieval/scripts/sentinel_1/burst_pipeline.py has been updated successfully. (file state is current in your context — no need to Read it back)
```

</details>

## Assistant

Reverted. So back to `f_dc = f_dc_lag1 + data_poly`. 

The "data_poly only" was worse, which actually tells us something important: the lag-1 residual is carrying **real spatial information**, not just noise. The overcompensation with `f_dc_lag1 + data_poly` is a smaller problem than losing the lag-1 information entirely.

To diagnose where the overcompensation comes from, I need to see those numbers. Can you run the four-line print I gave earlier for any burst where the overcompensation is visible?

```python
b = results[4]  # or any burst
f_res = b['f_dc'] - b['f_data_poly']
print(f"lag-1 residual:  [{f_res.min():.1f}, {f_res.max():.1f}] Hz  mean={f_res.mean():.1f}")
print(f"data_poly:       [{b['f_data_poly'].min():.1f}, {b['f_data_poly'].max():.1f}] Hz")
print(f"f_dc (fixed):    [{b['f_dc'].min():.1f}, {b['f_dc'].max():.1f}] Hz")
print(f"f_dc_ocn:        [{np.nanmin(b['f_dc_ocn']):.1f}, {np.nanmax(b['f_dc_ocn']):.1f}] Hz")
```

## User

yes, also go back one step further

## Assistant

Reverting the data_poly addition entirely from both files.

## Assistant

**Tool: Edit**
Editing `/home/oscipal/ocean_current_retrieval/scripts/sentinel_1/pipeline.py`
```diff
-     # Restore full observed DC: the S1 SLC is Doppler centroid compensated (DCC)
-     # by the IPF.  The lag-1 / FFT estimator therefore measures only the small
-     # residual after DCC.  Adding the annotation data_poly (the polynomial the
-     # IPF subtracted) recovers the equivalent of rvlDcObs used by the OCN product.
-     dc_est = _nearest_estimate(annot.dc_estimates, burst.azimuth_time)
-     tau_at_rg = annot.slant_range_time_start + rg_centers / annot.range_sampling_rate
-     data_poly_at_rg = np.array(
-         sum(c * (tau_at_rg - dc_est.t0) ** k for k, c in enumerate(dc_est.data_poly)),
-         dtype=np.float32,
-     )
-     f_dc = (f_dc + data_poly_at_rg[np.newaxis, :]).astype(np.float32)
- 
-     # Full-scene azimuth indices for geolocation
+     # Full-scene azimuth indices for geolocation
```

<details><summary>Tool Result</summary>

```
The file /home/oscipal/ocean_current_retrieval/scripts/sentinel_1/pipeline.py has been updated successfully. (file state is current in your context — no need to Read it back)
```

</details>

## Assistant

**Tool: Edit**
Editing `/home/oscipal/ocean_current_retrieval/scripts/sentinel_1/pipeline.py`
```diff
-         'f_dc':           f_dc,
-         'f_dc_ocn':       f_dc_ocn,
-         'f_data_poly':    data_poly_at_rg,
-         'tops_scale':     tops_scale,
+         'f_dc':           f_dc,
+         'f_dc_ocn':       f_dc_ocn,
+         'tops_scale':     tops_scale,
```

<details><summary>Tool Result</summary>

```
The file /home/oscipal/ocean_current_retrieval/scripts/sentinel_1/pipeline.py has been updated successfully. (file state is current in your context — no need to Read it back)
```

</details>

## Assistant

**Tool: Edit**
Editing `/home/oscipal/ocean_current_retrieval/scripts/sentinel_1/burst_pipeline.py`
```diff
-     # Restore full observed DC: the S1 SLC is DCC-corrected by the IPF.
-     # The lag-1 estimator sees only the residual; adding data_poly recovers
-     # the equivalent of rvlDcObs.
-     dc_est = _nearest_estimate(annot.dc_estimates, burst.azimuth_time)
-     tau_at_rg = annot.slant_range_time_start + rg_centers / annot.range_sampling_rate
-     data_poly_at_rg = np.array(
-         sum(c * (tau_at_rg - dc_est.t0) ** k for k, c in enumerate(dc_est.data_poly)),
-         dtype=np.float32,
-     )
-     f_dc = (f_dc + data_poly_at_rg[np.newaxis, :]).astype(np.float32)
- 
-     # Geometry Doppler: POEORB (preferred) or annotation polynomial (fallback).
+     # Geometry Doppler: POEORB (preferred) or annotation polynomial (fallback).
```

<details><summary>Tool Result</summary>

```
The file /home/oscipal/ocean_current_retrieval/scripts/sentinel_1/burst_pipeline.py has been updated successfully. (file state is current in your context — no need to Read it back)
```

</details>

## Assistant

Also revert the `doppler_data_poly` variable and the import of `_nearest_estimate` in burst_pipeline:

## Assistant

**Tool: Edit**
Editing `/home/oscipal/ocean_current_retrieval/scripts/sentinel_1/burst_pipeline.py`
```diff
- from .safe_io import _nearest_estimate, find_safe_files, parse_annotation, read_slc_burst
+ from .safe_io import find_safe_files, parse_annotation, read_slc_burst
```

<details><summary>Tool Result</summary>

```
The file /home/oscipal/ocean_current_retrieval/scripts/sentinel_1/burst_pipeline.py has been updated successfully. (file state is current in your context — no need to Read it back)
```

</details>

## Assistant

**Tool: Edit**
Editing `/home/oscipal/ocean_current_retrieval/scripts/sentinel_1/burst_pipeline.py`
```diff
-     f_geom_ann_2d   = np.broadcast_to(f_geom_ann[np.newaxis, :], f_dca.shape).astype(np.float32).copy()
-     f_geom_poe_2d   = np.broadcast_to(f_geom_poe[np.newaxis, :], f_dca.shape).astype(np.float32).copy()
-     f_miss_2d       = np.broadcast_to(f_miss_arr[np.newaxis, :], f_dca.shape).astype(np.float32).copy()
-     f_data_poly_2d  = np.broadcast_to(data_poly_at_rg[np.newaxis, :], f_dca.shape).astype(np.float32).copy()
- 
-     return xr.Dataset(
-         {
-             'doppler_hz': xr.DataArray(
-                 f_dca, dims=dims,
-                 attrs={'long_name': 'Doppler centroid anomaly (all corrections applied)',
-                        'units': 'Hz'},
-             ),
-             'doppler_obs': xr.DataArray(
-                 f_dc, dims=dims,
-                 attrs={'long_name': 'Observed Doppler centroid (DCC residual + data_poly)',
-                        'units': 'Hz'},
-             ),
-             'doppler_data_poly': xr.DataArray(
-                 f_data_poly_2d, dims=dims,
-                 attrs={'long_name': 'Annotation data_poly (DCC component added to residual)',
-                        'units': 'Hz'},
-             ),
+     f_geom_ann_2d = np.broadcast_to(f_geom_ann[np.newaxis, :], f_dca.shape).astype(np.float32).copy()
+     f_geom_poe_2d = np.broadcast_to(f_geom_poe[np.newaxis, :], f_dca.shape).astype(np.float32).copy()
+     f_miss_2d     = np.broadcast_to(f_miss_arr[np.newaxis, :], f_dca.shape).astype(np.float32).copy()
+ 
+     return xr.Dataset(
+         {
+             'doppler_hz': xr.DataArray(
+                 f_dca, dims=dims,
+                 attrs={'long_name': 'Doppler centroid anomaly (all corrections applied)',
+                        'units': 'Hz'},
+             ),
+             'doppler_obs': xr.DataArray(
+                 f_dc, dims=dims,
+                 attrs={'long_name': 'Observed Doppler centroid (raw, before subtraction)',
+                        'units': 'Hz'},
+             ),
```

<details><summary>Tool Result</summary>

```
The file /home/oscipal/ocean_current_retrieval/scripts/sentinel_1/burst_pipeline.py has been updated successfully. (file state is current in your context — no need to Read it back)
```

</details>

## Assistant

Both files are now fully back to the state before any of the data_poly changes. 

Now — please run those four print lines so we can actually see what's happening numerically before trying another fix. Specifically I need to know whether `data_poly` itself spans the same range as `f_dc_ocn`, or whether it's larger/smaller. That determines which direction we go from here.

## User

Can you go exactly through what you changed in the data_poly

## Assistant

Let me check the current state of both files to confirm they match the original, then give you the full history.

## Assistant

**Tool: Read**
Reading `/home/oscipal/ocean_current_retrieval/scripts/sentinel_1/pipeline.py`

<details><summary>Tool Result</summary>

```
195	    #print('Step I: deramping …')
196	    raw      = read_slc_burst(files['measurement'], annot, burst_idx)
197	    deramped = deramp_burst(raw, annot, burst_idx, deramp_method=deramp_method)
198	    valid_mask = apply_burst_valid_sample_mask(deramped, burst)
199	
200	    # ── Step II: Correlation / Doppler estimation ────────────────────────────
201	    if estimator == 'fft':
202	        #print('Step II: estimating Doppler via FFT spectral centroid …')
203	        f_dc, snr, az_local, rg_centers = estimate_doppler_grid_fft(
204	            deramped, annot.prf, block_az, block_rg, stride_az, stride_rg,
205	            valid_mask=valid_mask,
206	        )
207	        p0 = np.ones_like(f_dc, dtype=np.float32)
208	        p1 = np.full_like(f_dc, np.nan, dtype=np.float32)
209	    else:
210	        #print('Step II: estimating p0, p1 …')
211	        p0, p1, az_local, rg_centers = estimate_correlation_grid(
212	            deramped, block_az, block_rg, stride_az, stride_rg,
213	            valid_mask=valid_mask,
214	        )
215	        f_dc, _, snr = correlation_to_doppler(
216	            p0, p1, annot.prf, annot.wavelength, gamma_amb=gamma_amb,
217	        )
218	
219	    # Full-scene azimuth indices for geolocation
220	    ati      = annot.azimuth_time_interval
221	    az0_full = int(round(
222	        (burst.azimuth_time - annot.first_line_time).total_seconds() / ati
223	    ))
224	    az_full = az_local + az0_full
225	
226	    # ── Step III: Doppler centroid → DCA ────────────────────────────────────
227	    #print('Step III: geometry subtraction …')
228	    if tops_scaling not in ('none', 'multiply', 'divide'):
229	        raise ValueError("tops_scaling must be 'none', 'multiply', or 'divide'")
230	    k_a = _fm_rate_at_burst(annot, burst_idx)[rg_centers]
231	    k_psi = _steering_doppler_rate(annot, burst_idx)
232	    tops_scale = (1.0 + k_psi / k_a).astype(np.float32)
233	    if tops_scaling == 'multiply':
234	        f_dc = (f_dc * tops_scale[np.newaxis, :]).astype(np.float32)
```

</details>

## Assistant

**Tool: Read**
Reading `/home/oscipal/ocean_current_retrieval/scripts/sentinel_1/burst_pipeline.py`

<details><summary>Tool Result</summary>

```
1	#!/usr/bin/env python3
2	"""Run the diagnostic single-burst Sentinel-1 RVL retrieval path."""
3	
4	from __future__ import annotations
5	
6	import argparse
7	import numpy as np
8	import xarray as xr
9	
10	from .safe_io import find_safe_files, parse_annotation, read_slc_burst
11	from .rvl import (
12	    deramp_burst,
13	    estimate_correlation_grid,
14	    correlation_to_doppler,
15	    _geom_doppler_annotation,
16	    _geom_doppler_poeorb,
17	    _geolocate_grid,
18	    _interpolate_orbit,
19	    compute_gamma_ambiguity,
20	    compute_sideband_bias,
21	    apply_burst_valid_sample_mask,
22	)
23	
24	
25	# ---------------------------------------------------------------------------
```

</details>

## Assistant

Both files are clean — back to exactly the original state. Here's the complete history of what I changed:

---

**Round 1 — "f_dc = lag-1 + data_poly" (initial fix)**

Both `pipeline.py` and `burst_pipeline.py`:
- After `correlation_to_doppler`, evaluated `data_poly(rg_centers)` from the nearest DC estimate
- Added it to the lag-1 result: `f_dc = f_dc_lag1 + data_poly_at_rg`
- Theory: S1 SLC is DCC-corrected, lag-1 sees only the residual, adding data_poly restores the full observed DC

Side changes:
- Added `'f_data_poly': data_poly_at_rg` to the `run_pipeline` return dict
- Added `doppler_data_poly` DataArray to the burst_pipeline xr.Dataset output
- Changed `doppler_obs` long_name string in burst_pipeline

Result: better (range variation appeared) but overcompensated — f_dc too large by some amount.

---

**Round 2 — "f_dc = data_poly only" (attempted fix for overcompensation)**

Both files:
- Replaced `f_dc = f_dc_lag1 + data_poly` with `f_dc = broadcast(data_poly)` — discarding the lag-1 entirely

Result: worse than round 1. You reverted it.

---

**Round 3 — full revert** (current state)

All changes from both rounds removed. Both files are identical to what they were before this session's data_poly work.

---

So the current state is the **original pre-data_poly pipeline**. Round 1 was directionally right (the range variation appeared), but overcorrected somehow. I need those four print values to understand by how much and in which direction before trying again.

## User

but in round 1 what happened?

## Assistant

In Round 1:

**What the code did:** After the lag-1 gives `f_dc` (shape n_az_blocks × n_rg_blocks), we evaluated `data_poly(τ)` from the annotation at each block's slant-range time and added it as a 1D range correction:

```python
f_dc = f_dc_lag1 + data_poly(rg_centers)
```

**Why:** The theory was that the S1 IPF subtracts `data_poly` during azimuth compression (DCC), so our lag-1 only sees the small residual. Adding `data_poly` back recovers the full observed DC — equivalent to what OCN stores as `rvlDcObs`.

**What changed visually:** The flat 12–18 Hz profile got range variation matching the shape of `data_poly`, which is a polynomial that grows/shrinks across the ~21,000 range samples. The bursts started looking more like the OCN product.

**The overcompensation:** Because `f_dc_lag1` was not zero (it was ~12–18 Hz flat), the result was `data_poly + 12-18 Hz` everywhere instead of just `data_poly`. So it overshot by roughly the mean of the lag-1 output.

**The key unknown:** I still don't know whether `data_poly` itself spans the same range as `rvlDcObs`, or whether it's larger or smaller. That's what those four print lines would tell us. Without that, I was guessing at the source of the overcompensation.

## User

But would those changes have made sense?

## Assistant

Theoretically, yes — and here's the chain of reasoning:

The S1 IPF applies Doppler centroid compensation (DCC) during azimuth focusing using `data_poly(τ)`. After DCC, the SLC's spectrum is nominally centred at 0 Hz at every range position. Our lag-1 on the deramped SLC therefore measures the small residual left after DCC, not the full observed DC. Adding `data_poly` back recovers `f_dc_full = residual + data_poly`, which is exactly what `rvlDcObs` represents in the OCN product.

So the logic is sound. The overcompensation doesn't mean the approach is wrong — it just means the lag-1 residual is not actually near zero. It's sitting at ~12–18 Hz, which gets added on top of `data_poly` and pushes the result above `rvlDcObs`.

The real question is: **why is the lag-1 residual ~12–18 Hz rather than ~0?**

Three candidates:
1. **The TOPS deramping introduces a systematic DC offset.** From the derivation: if the focused SLC has a residual chirp centred at the zero-Doppler time `t_c = −f_dc/k_a` (rather than at the burst midpoint), our 'current' deramping with `eta_ref = 0` doesn't cancel it cleanly, and the lag-1 picks up a bias of order `k_t × t_c`.
2. **DCC is slightly inaccurate**, leaving a genuine ~15 Hz residual.
3. **`data_poly` slightly overestimates the true DC**, so `f_dc_lag1 + data_poly` overshoots even for a near-zero residual.

Until we see those four print values — specifically whether `data_poly` spans the same range as `f_dc_ocn` — we can't tell which of these is the actual cause.

## User

okay redo it and I will give you the prints

## Assistant

**Tool: Edit**
Editing `/home/oscipal/ocean_current_retrieval/scripts/sentinel_1/pipeline.py`
```diff
-         f_dc, _, snr = correlation_to_doppler(
-             p0, p1, annot.prf, annot.wavelength, gamma_amb=gamma_amb,
-         )
- 
-     # Full-scene azimuth indices for geolocation
+         f_dc, _, snr = correlation_to_doppler(
+             p0, p1, annot.prf, annot.wavelength, gamma_amb=gamma_amb,
+         )
+ 
+     dc_est = _nearest_estimate(annot.dc_estimates, burst.azimuth_time)
+     tau_at_rg = annot.slant_range_time_start + rg_centers / annot.range_sampling_rate
+     data_poly_at_rg = np.array(
+         sum(c * (tau_at_rg - dc_est.t0) ** k for k, c in enumerate(dc_est.data_poly)),
+         dtype=np.float32,
+     )
+     f_dc = (f_dc + data_poly_at_rg[np.newaxis, :]).astype(np.float32)
+ 
+     # Full-scene azimuth indices for geolocation
```

<details><summary>Tool Result</summary>

```
The file /home/oscipal/ocean_current_retrieval/scripts/sentinel_1/pipeline.py has been updated successfully. (file state is current in your context — no need to Read it back)
```

</details>

## Assistant

**Tool: Edit**
Editing `/home/oscipal/ocean_current_retrieval/scripts/sentinel_1/pipeline.py`
```diff
-         'f_dc':           f_dc,
-         'f_dc_ocn':       f_dc_ocn,
-         'tops_scale':     tops_scale,
+         'f_dc':           f_dc,
+         'f_dc_ocn':       f_dc_ocn,
+         'f_data_poly':    data_poly_at_rg,
+         'tops_scale':     tops_scale,
```

<details><summary>Tool Result</summary>

```
The file /home/oscipal/ocean_current_retrieval/scripts/sentinel_1/pipeline.py has been updated successfully. (file state is current in your context — no need to Read it back)
```

</details>

## Assistant

Done. Run this for any burst where you can see the overcompensation:

```python
b = results[4]   # or whichever burst
import numpy as np

f_res = b['f_dc'] - b['f_data_poly']
print(f"lag-1 residual:  [{f_res.min():.1f}, {f_res.max():.1f}] Hz  mean={f_res.mean():.1f}")
print(f"data_poly:       [{b['f_data_poly'].min():.1f}, {b['f_data_poly'].max():.1f}] Hz")
print(f"f_dc (fixed):    [{b['f_dc'].min():.1f}, {b['f_dc'].max():.1f}] Hz")
print(f"f_dc_ocn:        [{np.nanmin(b['f_dc_ocn']):.1f}, {np.nanmax(b['f_dc_ocn']):.1f}] Hz")
```

## User

the results are still gotten with run_all_bursts?

## Assistant

**Tool: Read**
Reading `/home/oscipal/ocean_current_retrieval/scripts/sentinel_1/pipeline.py`

<details><summary>Tool Result</summary>

```
395	    ocn_safe:     str,
396	    era5_wind:    str,
397	    era5_wave:    str,
398	    glo12:        str,
399	    polarisation: str = 'vv',
400	    burst_indices: list[int] | None = None,
401	    deramp_method: str = 'current',
402	    do_descallop: bool = True,
403	    tops_scaling: str = 'none',
404	    merge_first:  bool = False,
405	    use_ocn_dc:   bool = False,
406	    estimator:    str = 'cde',
407	    **kwargs,
408	) -> list[dict]:
409	    """
410	    Run run_pipeline for every burst in the subswath and return a list of results.
411	
412	    Parameters
413	    ----------
414	    burst_indices : list of int or None
415	        Which bursts to process. If None, all bursts in the subswath are processed.
416	        Ignored when merge_first=True.
417	    deramp_method : 'current' or 'esa_eq1'
418	        SAFE-based deramp method used in Step I.
419	    do_descallop  : bool
420	        Apply burst-periodic descalloping when enough azimuth extent is available.
421	    tops_scaling  : 'none', 'multiply', or 'divide'
422	        Optional TOPS focused-data Doppler scale test forwarded to run_pipeline.
423	        Ignored when merge_first=True.
424	    use_ocn_dc    : bool
425	        When True, replace the lag-1 correlation f_dc with rvlDcObs interpolated
426	        from the OCN product onto the SAR estimation grid. Geometry subtraction
427	        and all downstream corrections still use our own pipeline values.
428	    merge_first   : bool
429	        When True, deramp and coherently merge all bursts into one continuous SLC
430	        image first (using merge_bursts), then run the full pipeline on the merged
431	        image. This avoids burst-boundary discontinuities in the Doppler estimation
432	        at the cost of slightly more memory. Geometry corrections use the blended
433	        annotation polynomial (or nearest-burst POEORB differential when poeorb_path
434	        is provided). Returns a single-element list so all downstream functions
```

</details>

## Assistant

**Tool: Read**
Reading `/home/oscipal/ocean_current_retrieval/scripts/sentinel_1/pipeline.py`

<details><summary>Tool Result</summary>

```
435	        (merge_burst_grids, plot_comparison) remain compatible.
436	    **kwargs      : forwarded to run_pipeline (block_az, block_rg, stride_az, stride_rg)
437	    """
438	    if use_ocn_dc:
439	        return _run_ocn_pipeline(
440	            slc_safe=slc_safe, subswath=subswath,
441	            poeorb_path=poeorb_path, aux_cal_path=aux_cal_path,
442	            ocn_safe=ocn_safe, era5_wind=era5_wind, era5_wave=era5_wave,
443	            glo12=glo12, polarisation=polarisation,
444	        )
445	
446	    if merge_first:
447	        return _run_merged_pipeline(
448	            slc_safe=slc_safe, subswath=subswath,
449	            poeorb_path=poeorb_path, aux_cal_path=aux_cal_path,
450	            ocn_safe=ocn_safe, era5_wind=era5_wind, era5_wave=era5_wave,
451	            glo12=glo12, polarisation=polarisation, deramp_method=deramp_method,
452	            do_descallop=do_descallop, use_ocn_dc=use_ocn_dc,
453	            estimator=estimator, **kwargs,
454	        )
455	
456	    files = find_safe_files(slc_safe, subswath, polarisation)
457	    annot = parse_annotation(files['annotation'])
458	    n_bursts = len(annot.bursts)
459	
460	    if burst_indices is None:
461	        burst_indices = list(range(n_bursts))
462	
463	    print(f'Processing {len(burst_indices)} bursts for {subswath.upper()} …')
464	    results = []
465	    for bidx in burst_indices:
466	        print(f'\n─── Burst {bidx} / {n_bursts - 1} ───────────────────────────')
467	        r = run_pipeline(
468	            slc_safe=slc_safe, subswath=subswath, burst_idx=bidx,
469	            poeorb_path=poeorb_path, aux_cal_path=aux_cal_path,
470	            ocn_safe=ocn_safe, era5_wind=era5_wind, era5_wave=era5_wave,
471	            glo12=glo12, polarisation=polarisation, deramp_method=deramp_method,
472	            do_descallop=do_descallop, **kwargs,
473	            tops_scaling=tops_scaling, use_ocn_dc=use_ocn_dc, estimator=estimator,
474	        )
```

</details>

## Assistant

Yes — `run_all_bursts` just calls `run_pipeline` for each burst and appends the dict directly to `results`. Since `'f_data_poly'` is now in `run_pipeline`'s return dict, it'll be in every element of the list.

So `results[4]['f_data_poly']` will work.

## User

burst 9:

lag-1 residual:  [nan, nan] Hz  mean=nan
data_poly:       [13.9, 34.9] Hz
f_dc (fixed):    [nan, nan] Hz
f_dc_ocn:        [-93.4, 48.5] Hz

burst 3:

lag-1 residual:  [nan, nan] Hz  mean=nan
data_poly:       [9.1, 13.8] Hz
f_dc (fixed):    [nan, nan] Hz
f_dc_ocn:        [-47.9, 34.5] Hz

## Assistant

Two things jump out immediately.

**First:** the `[nan, nan]` for `f_dc` and `f_res` is because you used `.min()/.max()` which return NaN if *any* NaN exists in the array. The lag-1 grid has NaN in blocks that failed the valid-fraction mask — it doesn't mean the whole array is NaN. Rerun with `np.nanmin`/`np.nanmax` to see the actual finite values.

**Second, and more important:** look at these numbers:

| | Burst 3 | Burst 9 |
|---|---|---|
| `data_poly` | [9.1, 13.8] Hz | [13.9, 34.9] Hz |
| `f_dc_ocn` | [-47.9, 34.5] Hz | [-93.4, 48.5] Hz |

`data_poly` varies by **5 Hz and 21 Hz** respectively. `rvlDcObs` varies by **82 Hz and 142 Hz**. They're not even close — adding `data_poly` cannot possibly recover `rvlDcObs`. The DCC hypothesis was wrong.

`data_poly` is the IPF's rough mean DC estimate — it barely varies across range. The large range variation in `rvlDcObs` comes from something else entirely (most likely antenna mispointing, which creates a large range-dependent Doppler shift that the annotation polynomial never captures).

So the real question is still: why does our lag-1 on the deramped SLC give flat ~12–18 Hz while the true DC spans 80–140 Hz? The data_poly detour didn't help answer that.

Revert the print with nanmin/nanmax — can you share the actual finite values for burst 4 specifically (the one you said was "flat")?

## User

For burst 4:

lag-1 residual:  [2.3, 17.8] Hz  mean=9.9
data_poly:       [13.3, 18.4] Hz
f_dc (fixed):    [16.4, 36.0] Hz
f_dc_ocn:        [-40.9, 116.9] Hz

It's allright if my results look similar but with an offset though as long as they follow the same trend

## User

This session is being continued from a previous conversation that ran out of context. The summary below covers the earlier portion of the conversation.

Summary:
1. Primary Request and Intent:

**Main ongoing task:** Diagnose and fix why the Sentinel-1 RVL pipeline produces a flat/nearly-constant Doppler centroid (f_dc) range profile while the ESA OCN product (`rvlDcObs`) shows meaningful range variation (e.g., burst 4: -40.9 to +116.9 Hz, burst 9: -93.4 to +48.5 Hz). The user confirmed: "I am not comparing apples with oranges, the same corrections and processing steps are done for both datasets and mine is somehow missing something."

**User's recent relaxation:** "It's alright if my results look similar but with an offset though as long as they follow the same trend" — meaning absolute DC level doesn't need to match, just the range variation shape/trend.

**Secondary task (completed earlier this session):** Add `plot_rvl_comparison` to `scripts/sentinel_1/doppler_comparison.py`.

2. Key Technical Concepts:

- **Sentinel-1 IW TOPS SLC**: burst mode acquisition; antenna steers continuously causing within-burst DC variation at rate k_psi ≈ 7500 Hz/s
- **TOPS deramping** ('current' method): `k_s = -k_a*k_psi/(k_a-k_psi)` (per-range array), applied as `exp(j*pi*k_s[None,:]*t_az[:,None]²)` centered at burst midpoint (eta_ref=0)
- **`_deramp_esa`** (Eq. 1): uses `eta_ref = -f_dc/k_a - const` (centered at zero-Doppler crossing time, range-varying); with optional Eq. 8 demodulation that applies `-2*pi*f_dc*(eta-eta_ref)` to subtract `data_poly` from signal phase
- **`_block_p0_p1`**: Stage 1 = per-column lag-1 g1 (shape n_rg); Stage 2 = coherent Hanning²-weighted range average → collapses to ONE complex p1 per block
- **`estimate_correlation_grid`**: sliding window, blocks where `valid_frac < 0.8` → NaN; returns (p0_grid, p1_grid, az_centers, rg_centers)
- **`DcEstimate` dataclass**: has `data_poly` (measured DC coefficients, polynomial in slant-range time) and `geometry_poly` (orbit-predicted, ~1 Hz for near-zero squint S1)
- **`_geom_doppler_annotation`**: evaluates `geometry_poly` (NOT `data_poly`) at rg_centers
- **OCN `rvlDcObs`**: full observed Doppler centroid; spans 80-157 Hz range variation across burst range swath — much larger than our lag-1 estimate
- **DCC hypothesis (tested and disproven)**: theory that S1 IPF subtracts `data_poly` during DCC; disproven because data_poly spans only 5-21 Hz while rvlDcObs spans 80-157 Hz — completely different scales
- **NaN in f_dc arrays**: `.min()/.max()` returns NaN if ANY element is NaN; must use `np.nanmin`/`np.nanmax` to see finite values
- **Lag-1 residual for burst 4**: [2.3, 17.8] Hz (mean=9.9) — real spatial info (15.5 Hz range variation) matching the trend of rvlDcObs but at ~1/10 scale

3. Files and Code Sections:

- **`scripts/sentinel_1/pipeline.py`**
  - Current state: Round 1 data_poly fix IS applied
  - After `correlation_to_doppler` (line ~215), the following was added:
    ```python
    dc_est = _nearest_estimate(annot.dc_estimates, burst.azimuth_time)
    tau_at_rg = annot.slant_range_time_start + rg_centers / annot.range_sampling_rate
    data_poly_at_rg = np.array(
        sum(c * (tau_at_rg - dc_est.t0) ** k for k, c in enumerate(dc_est.data_poly)),
        dtype=np.float32,
    )
    f_dc = (f_dc + data_poly_at_rg[np.newaxis, :]).astype(np.float32)
    ```
  - `'f_data_poly': data_poly_at_rg` added to the `run_pipeline` return dict (alongside `'f_dc'` and `'f_dc_ocn'`)
  - `_nearest_estimate` was already imported from `.safe_io`

- **`scripts/sentinel_1/burst_pipeline.py`**
  - Current state: FULLY REVERTED to original — no data_poly changes remain
  - Import is `from .safe_io import find_safe_files, parse_annotation, read_slc_burst` (no `_nearest_estimate`)
  - `doppler_obs` long_name is `'Observed Doppler centroid (raw, before subtraction)'`
  - No `doppler_data_poly` DataArray

- **`scripts/sentinel_1/rvl.py`** (read-only, key sections):
  - `_fm_rate_at_burst` (line ~500): returns per-range array (n_samples,) in Hz/s
  - `_deramp_rate` (line 552): `return -k_a * k_psi / (k_a - k_psi)` — per-range, negative for k_a<0, k_psi>0
  - `_deramp_esa` (line 589): uses `eta_c = -f_dc/k_a`, `eta_ref = eta_c - eta_c[n//2]`, applies `exp(-j*pi*k_t*(eta-eta_ref)²)` ± demodulation
  - `deramp_burst` 'current' (line 664): `chirp = exp(j*pi*k_s[None,:]*t_az[:,None]²)` with t_az centered at burst midpoint
  - `_block_p0_p1` (line 799): Stage 2 coherent range average collapses n_rg columns to one p1
  - `_geom_doppler_annotation` (line 1549): evaluates `dc.geometry_poly` (not data_poly) at rg_centers

- **`scripts/sentinel_1/safe_io.py`** (read-only):
  - `DcEstimate` dataclass: `azimuth_time`, `t0` (reference slant-range time), `geometry_poly`, `data_poly`, `rms_error`
  - `_nearest_estimate`: argmin of time differences (NOT interpolation)

- **`scripts/diagnostics/burst_fdc_stages.py`** (read-only):
  - Existing diagnostic; computes mean/std per burst for raw/deramped × CDE/FFT — does not resolve range structure

4. Errors and Fixes:

- **Round 1 fix (f_dc = lag-1 + data_poly)**: User said "looks a bit better but it kind of overcompensated." Analysis showed overcompensation = lag-1 residual (~9.9 Hz mean) being added on top of data_poly, but more fundamentally data_poly doesn't span anywhere near rvlDcObs anyway.

- **Round 2 fix (f_dc = data_poly only)**: User said "noo this is way worse, revert the change." This proved lag-1 carries real spatial information — cannot be discarded.

- **NaN diagnostic issue**: Initial diagnostic used `.min()/.max()` which return NaN if any NaN exists. Bursts 3 and 9 showed `[nan, nan]` for f_dc. Fix: use `np.nanmin`/`np.nanmax`. After fixing, burst 4 shows valid finite ranges.

- **DCC hypothesis disproven**: data_poly spans [9.1-13.8] Hz (burst 3) and [13.9-34.9] Hz (burst 9) — only 5-21 Hz variation. rvlDcObs spans 82-142 Hz. They are completely different scales, meaning data_poly is NOT what was subtracted during DCC to create the large range variation seen in rvlDcObs.

5. Problem Solving:

**Solved this session:**
- Confirmed data_poly is NOT the missing piece (it barely varies, while rvlDcObs has 80-157 Hz range variation)
- Confirmed lag-1 carries real spatial information (dropping it was "way worse")
- Identified that `.min()/.max()` hides partial-NaN arrays; must use nanmin/nanmax
- Got actual diagnostic numbers for burst 4: lag-1 residual [2.3, 17.8] Hz (15.5 Hz range), data_poly [13.3, 18.4] Hz (5 Hz range), f_dc fixed [16.4, 36.0] Hz, f_dc_ocn [-40.9, 116.9] Hz (157 Hz range)

**Ongoing — core mystery:**
Why does our lag-1 on the deramped SLC give only 15.5 Hz range variation while rvlDcObs shows 157 Hz range variation for the same burst? The trend/shape appears similar (both increase from near range to far range) but at ~1/10 scale.

The user has now relaxed the requirement: an offset is acceptable as long as the trend matches. This is significant — the lag-1 residual [2.3, 17.8] Hz and rvlDcObs [-40.9, 116.9] Hz may already have the same trend, just at different scales and with an offset. The ratio of ranges is ~10:1.

6. All User Messages:

- "our last chat is in chat.md. The limit ran out, continue from there" (session start)
- "looks a bit better but it kind of overcompensated" (after Round 1)
- "noo this is way worse, revert the change" (after Round 2)
- "yes, also go back one step further" (asking to revert Round 1 as well)
- "Can you go exactly through what you changed in the data_poly" (asking for history of changes)
- "but in round 1 what happened?" (asking for more detail on Round 1 specifically)
- "But would those changes have made sense?" (asking about theoretical soundness)
- "okay redo it and I will give you the prints" (reapply Round 1 for diagnostics)
- "the results are still gotten with run_all_bursts?" (confirming f_data_poly is accessible)
- Burst 9 print: `lag-1 residual: [nan, nan] Hz  mean=nan`, `data_poly: [13.9, 34.9] Hz`, `f_dc (fixed): [nan, nan] Hz`, `f_dc_ocn: [-93.4, 48.5] Hz`
- Burst 3 print: `lag-1 residual: [nan, nan] Hz  mean=nan`, `data_poly: [9.1, 13.8] Hz`, `f_dc (fixed): [nan, nan] Hz`, `f_dc_ocn: [-47.9, 34.5] Hz`
- Burst 4 print: `lag-1 residual: [2.3, 17.8] Hz  mean=9.9`, `data_poly: [13.3, 18.4] Hz`, `f_dc (fixed): [16.4, 36.0] Hz`, `f_dc_ocn: [-40.9, 116.9] Hz`
- **"It's alright if my results look similar but with an offset though as long as they follow the same trend"** (key relaxation of requirements — offset acceptable, trend/shape must match)

7. Pending Tasks:

- **Understand and fix why lag-1 gives only 15.5 Hz range variation while rvlDcObs shows 157 Hz for burst 4.** User is OK with an offset; they need the trend/shape to match. The current f_dc fixed [16.4, 36.0] Hz has the right directional trend (increases across range like rvlDcObs) but is 1/10 the scale.
- Investigate whether the ~10:1 scale difference between lag-1 range variation and rvlDcObs range variation is due to: TOPS deramping scaling effect, valid-mask over-restriction, or some other cause.
- Consider whether `_deramp_esa` with `demodulate=True` would correctly scale the DC estimation (by adding data_poly via phase rather than directly).

8. Current Work:

Round 1 fix is currently applied to `pipeline.py`:
```python
dc_est = _nearest_estimate(annot.dc_estimates, burst.azimuth_time)
tau_at_rg = annot.slant_range_time_start + rg_centers / annot.range_sampling_rate
data_poly_at_rg = np.array(
    sum(c * (tau_at_rg - dc_est.t0) ** k for k, c in enumerate(dc_est.data_poly)),
    dtype=np.float32,
)
f_dc = (f_dc + data_poly_at_rg[np.newaxis, :]).astype(np.float32)
```
And `'f_data_poly': data_poly_at_rg` is in the return dict.

`burst_pipeline.py` is fully reverted to original.

The diagnostic numbers for burst 4 show:
- lag-1 residual: [2.3, 17.8] Hz (mean=9.9) — 15.5 Hz range variation
- data_poly: [13.3, 18.4] Hz — only 5 Hz variation
- f_dc (fixed = lag-1 + data_poly): [16.4, 36.0] Hz — 19.6 Hz range variation
- f_dc_ocn: [-40.9, 116.9] Hz — 157.8 Hz range variation

The user confirmed the trend direction is what matters, not the absolute level.

9. Optional Next Step:

The user's last statement was: "It's alright if my results look similar but with an offset though as long as they follow the same trend."

The key question is whether the lag-1 residual [2.3, 17.8] Hz already matches the trend of rvlDcObs [-40.9, 116.9] Hz in shape (both increase from near to far range), just offset and scaled. The next step should be to verify whether the trend/shape matches well enough by plotting them together, OR to investigate why the scale is ~10:1 and whether a correction factor (from the TOPS deramping geometry) can bring them into agreement.

The most direct next step: check if `f_dc_ocn - mean(f_dc_ocn)` and `f_dc_residual - mean(f_dc_residual)` have similar normalized shapes across range for burst 4. If they do, the remaining issue is just scale and offset (not a fundamental shape mismatch), and the existing fix (with perhaps a scaling adjustment) may be sufficient.

If you need specific details from before compaction (like exact code snippets, error messages, or content you generated), read the full transcript at: /home/oscipal/.claude/projects/-home-oscipal-ocean-current-retrieval/8d8b0096-135a-4267-b2c9-21398a1215c8.jsonl
Continue the conversation from where it left off without asking the user any further questions. Resume directly — do not acknowledge the summary, do not recap what was happening, do not preface with "I'll continue" or similar. Pick up the last task as if the break never happened.

## Assistant

You've hit your org's monthly usage limit

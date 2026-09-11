import fs from "node:fs/promises";
import path from "node:path";
import { pathToFileURL } from "node:url";

const [inputPath, outputPath, previewDir, artifactToolPath] = process.argv.slice(2);
if (!inputPath || !outputPath || !previewDir || !artifactToolPath) {
  throw new Error("Usage: node build_boundary_segment_error_workbook.mjs INPUT_JSON OUTPUT_XLSX PREVIEW_DIR ARTIFACT_TOOL_MJS");
}
const { FileBlob, SpreadsheetFile, Workbook } = await import(pathToFileURL(artifactToolPath).href);
const payload = JSON.parse(await fs.readFile(inputPath, "utf8"));
const wb = Workbook.create();
const font = "Arial";
const dark = "#203864";
const lightBlue = "#D9EAF7";
const lightRed = "#FCE4D6";
const lightAmber = "#FFF2CC";

const summarySheet = wb.worksheets.add("Summary");
const comparisonSheet = wb.worksheets.add("Decoder comparison");
const calibratedSplitSheet = wb.worksheets.add("Calibrated split");
const calibratedClassSheet = wb.worksheets.add("Calibrated classes");
const calibratedRunSheet = wb.worksheets.add("Calibrated runs");
const calibratedFactorSheet = wb.worksheets.add("Calibrated factors");
const calibratedHardSheet = wb.worksheets.add("Calibrated hard");
const beforeAfterSheet = wb.worksheets.add("Before-after GT");
const splitSheet = wb.worksheets.add("Split summary");
const testClassSheet = wb.worksheets.add("Test classes");
const allClassSheet = wb.worksheets.add("All class summary");
const runSheet = wb.worksheets.add("Test runs");
const factorSheet = wb.worksheets.add("Duration and gap");
const hardSheet = wb.worksheets.add("Hard segments");
const allSheet = wb.worksheets.add("All GT segments");

function columnName(n) {
  let result = "";
  while (n > 0) {
    n -= 1;
    result = String.fromCharCode(65 + (n % 26)) + result;
    n = Math.floor(n / 26);
  }
  return result;
}

function titleBlock(sheet, title, subtitle, lastCol) {
  sheet.showGridLines = false;
  sheet.getRange("A1").values = [[title]];
  sheet.getRange("A1").format.font = { name: font, size: 15, bold: true, color: "#1F1F1F" };
  sheet.getRange("A2").values = [[subtitle]];
  sheet.getRange(`A2:${lastCol}2`).format.font = { name: font, size: 10, italic: true, color: "#666666" };
  sheet.getRange(`A1:${lastCol}1`).format.borders = { bottom: { style: "thin", color: "#A6A6A6" } };
}

function writeSheet(sheet, title, subtitle, columns, rows, tableName, widths = {}) {
  const lastCol = columnName(columns.length);
  titleBlock(sheet, title, subtitle, lastCol);
  sheet.getRange(`A4:${lastCol}4`).values = [columns.map((column) => column[0])];
  const matrix = rows.map((row) => columns.map((column) => row[column[1]] ?? null));
  if (matrix.length) sheet.getRange("A5").write(matrix);
  const endRow = 4 + matrix.length;
  sheet.getRange(`A4:${lastCol}4`).format = {
    fill: dark,
    font: { name: font, size: 9, bold: true, color: "#FFFFFF" },
    horizontalAlignment: "center",
    verticalAlignment: "center",
    wrapText: true,
    borders: { insideVertical: { style: "thin", color: "#FFFFFF" } },
  };
  if (matrix.length) {
    sheet.getRange(`A5:${lastCol}${endRow}`).format.font = { name: font, size: 9, color: "#1F1F1F" };
    sheet.getRange(`A5:${lastCol}${endRow}`).format.verticalAlignment = "top";
    const table = sheet.tables.add(`A4:${lastCol}${endRow}`, true, tableName);
    table.style = "TableStyleMedium2";
  }
  sheet.freezePanes.freezeRows(4);
  for (const [col, width] of Object.entries(widths)) sheet.getRange(`${col}:${col}`).format.columnWidth = width;
  return { endRow, lastCol };
}

const summaryColumns = [
  ["Split", "split"], ["GT segments", "segments"], ["Start miss ±5", "start_miss_rate_5"],
  ["End miss ±5", "end_miss_rate_5"], ["Any boundary miss ±5", "any_boundary_miss_rate_5"],
  ["Segment miss IoU 0.5", "segment_miss_rate_iou50"], ["Hard error", "hard_error_rate"],
];
const splitInfo = writeSheet(splitSheet, "Errors by data split", "Epoch-4 checkpoint; train is an in-sample replay, not a per-epoch trace.", summaryColumns, payload.split_summary, "SplitSummaryTable", { A: 18, B: 14, C: 16, D: 16, E: 23, F: 23, G: 16 });
splitSheet.getRange(`C5:G${splitInfo.endRow}`).format.numberFormat = "0.00%";
const calibratedSplitInfo = writeSheet(calibratedSplitSheet, "Validation-calibrated errors by data split", "Calibration parameters were selected only on 12 training-side validation runs.", summaryColumns, payload.calibrated_split_summary, "CalibratedSplitTable", { A: 18, B: 14, C: 16, D: 16, E: 23, F: 23, G: 16 });
calibratedSplitSheet.getRange(`C5:G${calibratedSplitInfo.endRow}`).format.numberFormat = "0.00%";

const classColumns = [
  ["Class", "class_label"], ["Action", "action"], ["Object", "object"], ["Segments", "segments"],
  ["Start miss ±3", "start_miss_rate_3"], ["Start miss ±5", "start_miss_rate_5"], ["Start miss ±10", "start_miss_rate_10"],
  ["End miss ±3", "end_miss_rate_3"], ["End miss ±5", "end_miss_rate_5"], ["End miss ±10", "end_miss_rate_10"],
  ["Any boundary miss ±5", "any_boundary_miss_rate_5"], ["Segment miss IoU 0.5", "segment_miss_rate_iou50"],
  ["Hard error", "hard_error_rate"], ["Median duration s", "median_duration_seconds"],
];
const testClassRows = [...payload.test_class_summary].sort((a, b) => b.segment_miss_rate_iou50 - a.segment_miss_rate_iou50 || b.any_boundary_miss_rate_5 - a.any_boundary_miss_rate_5 || b.segments - a.segments);
const testClassInfo = writeSheet(testClassSheet, "Held-out A errors by action-object class", "All test classes; use segment count with rates when judging stability.", classColumns, testClassRows, "TestClassTable", { A: 35, B: 20, C: 24, D: 12, E: 15, F: 15, G: 16, H: 15, I: 15, J: 16, K: 22, L: 23, M: 14, N: 18 });
testClassSheet.getRange(`E5:M${testClassInfo.endRow}`).format.numberFormat = "0.00%";
testClassSheet.getRange(`N5:N${testClassInfo.endRow}`).format.numberFormat = "0.00";
const calibratedClassRows = [...payload.calibrated_test_class_summary].sort((a, b) => b.segment_miss_rate_iou50 - a.segment_miss_rate_iou50 || b.any_boundary_miss_rate_5 - a.any_boundary_miss_rate_5 || b.segments - a.segments);
const calibratedClassInfo = writeSheet(calibratedClassSheet, "Validation-calibrated held-out A errors by class", "Primary class analysis after validation-only decoder calibration.", classColumns, calibratedClassRows, "CalibratedClassTable", { A: 35, B: 20, C: 24, D: 12, E: 15, F: 15, G: 16, H: 15, I: 15, J: 16, K: 22, L: 23, M: 14, N: 18 });
calibratedClassSheet.getRange(`E5:M${calibratedClassInfo.endRow}`).format.numberFormat = "0.00%";
calibratedClassSheet.getRange(`N5:N${calibratedClassInfo.endRow}`).format.numberFormat = "0.00";

const allClassColumns = [["Split", "split"], ...classColumns];
const allClassInfo = writeSheet(allClassSheet, "Class errors in every split", "Train, validation, test-normal and test-fault remain separate for comparison.", allClassColumns, payload.class_summary, "AllClassTable", { A: 18, B: 35, C: 20, D: 24, E: 12, F: 15, G: 15, H: 16, I: 15, J: 15, K: 16, L: 22, M: 23, N: 14, O: 18 });
allClassSheet.getRange(`F5:N${allClassInfo.endRow}`).format.numberFormat = "0.00%";
allClassSheet.getRange(`O5:O${allClassInfo.endRow}`).format.numberFormat = "0.00";

const runColumns = [
  ["Run", "sample_name"], ["Split", "split"], ["GT segments", "segments"], ["Start miss ±5", "start_miss_rate_5"],
  ["End miss ±5", "end_miss_rate_5"], ["Any boundary miss ±5", "any_boundary_miss_rate_5"],
  ["Segment miss IoU 0.5", "segment_miss_rate_iou50"], ["Hard error", "hard_error_rate"],
  ["Mean abs nearest start error", "mean_abs_nearest_start_error_frames"], ["Mean abs nearest end error", "mean_abs_nearest_end_error_frames"],
];
const runInfo = writeSheet(runSheet, "Held-out A errors by run", "Normal and fault runs are kept separate in the Split column.", runColumns, payload.test_run_summary, "TestRunTable", { A: 22, B: 16, C: 14, D: 16, E: 16, F: 22, G: 23, H: 14, I: 25, J: 25 });
runSheet.getRange(`D5:H${runInfo.endRow}`).format.numberFormat = "0.00%";
runSheet.getRange(`I5:J${runInfo.endRow}`).format.numberFormat = "0.00";
const calibratedRunInfo = writeSheet(calibratedRunSheet, "Validation-calibrated held-out A errors by run", "Normal and fault runs remain separate in the Split column.", runColumns, payload.calibrated_test_run_summary, "CalibratedRunTable", { A: 22, B: 16, C: 14, D: 16, E: 16, F: 22, G: 23, H: 14, I: 25, J: 25 });
calibratedRunSheet.getRange(`D5:H${calibratedRunInfo.endRow}`).format.numberFormat = "0.00%";
calibratedRunSheet.getRange(`I5:J${calibratedRunInfo.endRow}`).format.numberFormat = "0.00";

const factorRows = payload.duration_summary.map((row) => ({ factor: "duration", group: row.duration_bin, ...row })).concat(payload.gap_summary.map((row) => ({ factor: "nearest gap", group: row.gap_bin, ...row })));
const factorColumns = [
  ["Factor", "factor"], ["Group", "group"], ["Segments", "segments"], ["Start miss ±5", "start_miss_rate_5"],
  ["End miss ±5", "end_miss_rate_5"], ["Any boundary miss ±5", "any_boundary_miss_rate_5"],
  ["Segment miss IoU 0.5", "segment_miss_rate_iou50"], ["Hard error", "hard_error_rate"],
];
const factorInfo = writeSheet(factorSheet, "Errors by duration and nearest background gap", "Nearest gap is the smaller of the background steps before and after each GT segment.", factorColumns, factorRows, "FactorSummaryTable", { A: 18, B: 18, C: 13, D: 16, E: 16, F: 22, G: 23, H: 14 });
factorSheet.getRange(`D5:H${factorInfo.endRow}`).format.numberFormat = "0.00%";
const calibratedFactorRows = payload.calibrated_duration_summary.map((row) => ({ factor: "duration", group: row.duration_bin, ...row })).concat(payload.calibrated_gap_summary.map((row) => ({ factor: "nearest gap", group: row.gap_bin, ...row })));
const calibratedFactorInfo = writeSheet(calibratedFactorSheet, "Validation-calibrated errors by duration and gap", "Nearest gap is the smaller of the background steps before and after each GT segment.", factorColumns, calibratedFactorRows, "CalibratedFactorTable", { A: 18, B: 18, C: 13, D: 16, E: 16, F: 22, G: 23, H: 14 });
calibratedFactorSheet.getRange(`D5:H${calibratedFactorInfo.endRow}`).format.numberFormat = "0.00%";

const detailColumns = [
  ["Split", "split"], ["Phase", "phase"], ["Run", "sample_name"], ["Participant", "participant"], ["Source run", "source_run"],
  ["GT ordinal", "gt_segment_ordinal"], ["Segment no", "segment_no"], ["Action", "action"], ["Object", "object"], ["Class", "class_label"],
  ["GT start timestamp", "gt_start_timestamp"], ["GT end timestamp", "gt_end_timestamp"], ["GT start frame", "gt_start_original_frame_idx"], ["GT end frame", "gt_end_original_frame_idx"],
  ["GT start frame name", "gt_start_frame_name"], ["GT end frame name", "gt_end_frame_name"], ["GT start frame path", "gt_start_frame_path"], ["GT end frame path", "gt_end_frame_path"],
  ["Duration frames", "duration_frames"], ["Duration s", "duration_seconds"], ["Gap before", "background_gap_before_steps"], ["Gap after", "background_gap_after_steps"], ["Nearest gap", "nearest_background_gap_steps"],
  ["Start P at GT", "start_probability_at_gt"], ["End P at GT", "end_probability_at_gt"], ["Action P at GT start", "action_probability_at_gt_start"], ["Action P at GT end", "action_probability_at_gt_end"],
  ["Start local max r2", "start_probability_local_max_radius2"], ["End local max r2", "end_probability_local_max_radius2"], ["Start miss ±3", "start_miss_3"], ["Start miss ±5", "start_miss_5"], ["Start miss ±10", "start_miss_10"],
  ["End miss ±3", "end_miss_3"], ["End miss ±5", "end_miss_5"], ["End miss ±10", "end_miss_10"], ["Any boundary miss ±5", "any_boundary_miss_5"],
  ["Nearest start error frames", "nearest_start_error_frames"], ["Nearest end error frames", "nearest_end_error_frames"],
  ["Segment matched IoU 0.5", "segment_matched_iou50"], ["Segment IoU", "segment_iou"], ["Segment miss IoU 0.5", "segment_miss_iou50"],
  ["Matched pred start frame", "matched_predicted_start_original_frame_idx"], ["Matched pred end frame", "matched_predicted_end_original_frame_idx"],
  ["Error type", "error_type"], ["Severity", "severity_score"], ["Hard error", "hard_error"], ["Duration bin", "duration_bin"], ["Gap bin", "gap_bin"],
  ["Annotation file", "annotation_file"], ["Feature cache", "feature_cache"],
];
const widths = { A: 15, B: 16, C: 21, D: 12, E: 12, F: 12, G: 12, H: 24, I: 24, J: 34, K: 24, L: 24, M: 14, N: 14, O: 26, P: 26, Q: 66, R: 66, S: 15, T: 13, U: 12, V: 12, W: 12, X: 15, Y: 15, Z: 17, AA: 17, AB: 16, AC: 16, AD: 14, AE: 14, AF: 15, AG: 14, AH: 14, AI: 15, AJ: 20, AK: 21, AL: 19, AM: 14, AN: 19, AO: 21, AP: 21, AQ: 21, AR: 42, AS: 11, AT: 12, AU: 14, AV: 12, AW: 60, AX: 60 };
const hardInfo = writeSheet(hardSheet, "GT segments requiring review", "Hard error: start or end misses ±10 frames, or no one-to-one predicted segment reaches IoU 0.5.", detailColumns, payload.hard_segments, "HardSegmentsTable", widths);
const allInfo = writeSheet(allSheet, "All audited GT segments", "Every GT action segment in train, validation, test-normal and test-fault.", detailColumns, payload.all_segments, "AllSegmentsTable", widths);
for (const [sheet, info] of [[hardSheet, hardInfo], [allSheet, allInfo]]) {
  sheet.getRange(`X5:AC${info.endRow}`).format.numberFormat = "0.0000";
  sheet.getRange(`AN5:AN${info.endRow}`).format.numberFormat = "0.0000";
  sheet.getRange(`AR5:AR${info.endRow}`).conditionalFormats.add("containsText", { text: "missed", format: { fill: lightRed } });
  sheet.getRange(`AT5:AT${info.endRow}`).conditionalFormats.add("cellIs", { operator: "equal", formula: 1, format: { fill: lightAmber } });
}

const comparisonColumns = [
  ["Split", "split"], ["GT segments", "gt_segments"], ["Original pred", "original_predicted_segments"], ["Calibrated pred", "calibrated_predicted_segments"],
  ["Pred reduction", "predicted_segment_reduction_rate"], ["Original matches", "original_segment_matches_iou50"], ["Calibrated matches", "calibrated_segment_matches_iou50"],
  ["Original Seg P", "original_segment_precision_iou50"], ["Calibrated Seg P", "calibrated_segment_precision_iou50"], ["Original Seg R", "original_segment_recall_iou50"], ["Calibrated Seg R", "calibrated_segment_recall_iou50"],
  ["Original Seg F1", "original_segment_f1_iou50"], ["Calibrated Seg F1", "calibrated_segment_f1_iou50"], ["F1 delta pp", "segment_f1_delta_pp"],
  ["Original start miss ±5", "original_start_miss_rate_5"], ["Calibrated start miss ±5", "calibrated_start_miss_rate_5"], ["Start miss delta pp", "start_miss_delta_pp"],
  ["Original end miss ±5", "original_end_miss_rate_5"], ["Calibrated end miss ±5", "calibrated_end_miss_rate_5"], ["End miss delta pp", "end_miss_delta_pp"],
  ["Original segment miss", "original_segment_miss_rate_iou50"], ["Calibrated segment miss", "calibrated_segment_miss_rate_iou50"], ["Segment miss delta pp", "segment_miss_delta_pp"],
  ["Original hard error", "original_hard_error_rate"], ["Calibrated hard error", "calibrated_hard_error_rate"], ["Hard delta pp", "hard_error_delta_pp"],
  ["Resolved hard", "hard_errors_resolved"], ["New hard", "new_hard_errors"],
];
const comparisonInfo = writeSheet(comparisonSheet, "Original versus validation-calibrated decoder", "Same GT and one-to-one matching. Percentage-point deltas are calibrated minus original; negative miss deltas are improvements.", comparisonColumns, payload.decoder_comparison, "DecoderComparisonTable", { A: 18, B: 13, C: 14, D: 16, E: 15, F: 15, G: 17, H: 15, I: 17, J: 15, K: 17, L: 15, M: 17, N: 13, O: 20, P: 22, Q: 17, R: 19, S: 21, T: 16, U: 19, V: 21, W: 18, X: 18, Y: 20, Z: 14, AA: 14, AB: 12 });
comparisonSheet.getRange(`E5:E${comparisonInfo.endRow}`).format.numberFormat = "0.00%";
for (const range of [`H5:M${comparisonInfo.endRow}`, `O5:P${comparisonInfo.endRow}`, `R5:S${comparisonInfo.endRow}`, `U5:V${comparisonInfo.endRow}`, `X5:Y${comparisonInfo.endRow}`]) comparisonSheet.getRange(range).format.numberFormat = "0.00%";
comparisonSheet.getRange(`N5:N${comparisonInfo.endRow}`).format.numberFormat = "0.00";
comparisonSheet.getRange(`Q5:Q${comparisonInfo.endRow}`).format.numberFormat = "0.00";
comparisonSheet.getRange(`T5:T${comparisonInfo.endRow}`).format.numberFormat = "0.00";
comparisonSheet.getRange(`W5:W${comparisonInfo.endRow}`).format.numberFormat = "0.00";
comparisonSheet.getRange(`Z5:Z${comparisonInfo.endRow}`).format.numberFormat = "0.00";

const transitionColumns = [
  ["Split", "split"], ["Phase", "phase"], ["Run", "sample_name"], ["GT ordinal", "gt_segment_ordinal"], ["Segment no", "segment_no"],
  ["Action", "action"], ["Object", "object"], ["GT start timestamp", "gt_start_timestamp"], ["GT end timestamp", "gt_end_timestamp"],
  ["GT start frame path", "gt_start_frame_path"], ["GT end frame path", "gt_end_frame_path"], ["Duration s", "duration_seconds"], ["Nearest gap", "nearest_background_gap_steps"],
  ["Original start miss ±5", "start_miss_5"], ["Original end miss ±5", "end_miss_5"], ["Original segment miss", "segment_miss_iou50"], ["Original hard", "hard_error"],
  ["Original error type", "error_type"], ["Original nearest start error", "nearest_start_error_frames"], ["Original nearest end error", "nearest_end_error_frames"], ["Original IoU", "segment_iou"],
  ["Calibrated start miss ±5", "calibrated_start_miss_5"], ["Calibrated end miss ±5", "calibrated_end_miss_5"], ["Calibrated segment miss", "calibrated_segment_miss_iou50"], ["Calibrated hard", "calibrated_hard_error"],
  ["Calibrated error type", "calibrated_error_type"], ["Calibrated nearest start error", "calibrated_nearest_start_error_frames"], ["Calibrated nearest end error", "calibrated_nearest_end_error_frames"], ["Calibrated IoU", "calibrated_segment_iou"],
  ["Matched calibrated start frame", "calibrated_matched_predicted_start_original_frame_idx"], ["Matched calibrated end frame", "calibrated_matched_predicted_end_original_frame_idx"], ["Merged components", "calibrated_matched_merged_components"],
  ["Segment fixed", "calibrated_segment_fixed"], ["Segment became miss", "calibrated_segment_became_miss"], ["Hard resolved", "calibrated_hard_resolved"], ["New hard", "calibrated_hard_new"], ["Calibration outcome", "calibration_outcome"],
  ["Calibrated severity", "calibrated_severity_score"], ["Annotation file", "annotation_file"], ["Feature cache", "feature_cache"],
];
const transitionWidths = { A: 15, B: 16, C: 21, D: 12, E: 12, F: 24, G: 24, H: 24, I: 24, J: 66, K: 66, L: 13, M: 12, N: 20, O: 19, P: 19, Q: 14, R: 42, S: 21, T: 21, U: 13, V: 22, W: 21, X: 21, Y: 16, Z: 42, AA: 23, AB: 23, AC: 15, AD: 27, AE: 27, AF: 17, AG: 14, AH: 18, AI: 15, AJ: 12, AK: 27, AL: 17, AM: 60, AN: 60 };
const calibratedHardInfo = writeSheet(calibratedHardSheet, "GT segments requiring review after calibration", "Primary manual-review list. Filter Calibration outcome for persistent_hard_error or new_hard_error.", transitionColumns, payload.calibrated_hard_segments, "CalibratedHardTable", transitionWidths);
const beforeAfterInfo = writeSheet(beforeAfterSheet, "Per-GT original-to-calibrated comparison", "Every GT segment; original and calibrated results use identical one-to-one matching rules.", transitionColumns, payload.all_segments, "BeforeAfterTable", transitionWidths);
for (const [sheet, info] of [[calibratedHardSheet, calibratedHardInfo], [beforeAfterSheet, beforeAfterInfo]]) {
  sheet.getRange(`L5:L${info.endRow}`).format.numberFormat = "0.00";
  sheet.getRange(`U5:U${info.endRow}`).format.numberFormat = "0.0000";
  sheet.getRange(`AC5:AC${info.endRow}`).format.numberFormat = "0.0000";
  sheet.getRange(`AK5:AK${info.endRow}`).conditionalFormats.add("containsText", { text: "persistent", format: { fill: lightRed } });
  sheet.getRange(`AK5:AK${info.endRow}`).conditionalFormats.add("containsText", { text: "new_hard", format: { fill: lightAmber } });
  sheet.getRange(`AK5:AK${info.endRow}`).conditionalFormats.add("containsText", { text: "resolved", format: { fill: "#E2F0D9" } });
}

titleBlock(summarySheet, "Boundary segment errors", "A-as-test, seed 1, all-runs, epoch-4 checkpoint; original baseline plus validation-only calibrated decoder.", "N");
summarySheet.getRange("A4:G4").values = [["Calibrated split", "GT segments", "Start miss ±5", "End miss ±5", "Any boundary miss ±5", "Segment miss IoU 0.5", "Hard error"]];
for (let i = 0; i < payload.calibrated_split_summary.length; i += 1) {
  const row = 5 + i;
  for (let col = 0; col < 7; col += 1) summarySheet.getCell(row - 1, col).formulas = [[`='Calibrated split'!${columnName(col + 1)}${row}`]];
}
summarySheet.getRange("C5:G8").format.numberFormat = "0.00%";
summarySheet.getRange("A11:D11").values = [["Calibrated test class", "Segments", "Start miss ±5", "Segment miss IoU 0.5"]];
const topClasses = calibratedClassRows.slice(0, 12);
summarySheet.getRange("A12:D23").values = topClasses.map((row) => [row.class_label, row.segments, row.start_miss_rate_5, row.segment_miss_rate_iou50]);
summarySheet.getRange("C12:D23").format.numberFormat = "0.00%";
for (const range of ["A4:G4", "A11:D11"]) summarySheet.getRange(range).format = { fill: dark, font: { name: font, size: 9, bold: true, color: "#FFFFFF" }, horizontalAlignment: "center", wrapText: true };
summarySheet.getRange("A5:G8").format.font = { name: font, size: 10 };
summarySheet.getRange("A12:D23").format.font = { name: font, size: 9 };
summarySheet.getRange("A26:E26").values = [["Split", "Original pred", "Calibrated pred", "Original Seg F1", "Calibrated Seg F1"]];
summarySheet.getRange("A27:E30").values = payload.decoder_comparison.map((row) => [row.split, row.original_predicted_segments, row.calibrated_predicted_segments, row.original_segment_f1_iou50, row.calibrated_segment_f1_iou50]);
summarySheet.getRange("D27:E30").format.numberFormat = "0.00%";
summarySheet.getRange("A26:E26").format = { fill: dark, font: { name: font, size: 9, bold: true, color: "#FFFFFF" }, horizontalAlignment: "center", wrapText: true };
summarySheet.getRange("A27:E30").format.font = { name: font, size: 9 };
summarySheet.getRange("A32:E34").values = [["Calibration", "start=0.85", "end=0.85", "action=0.65", "merge gap≤3"], ["Selection", "12 validation runs only", null, null, null], ["Review", "Use Calibrated hard", "then Before-after GT", null, null]];
summarySheet.getRange("A32:A34").format.font = { name: font, size: 9, bold: true, color: dark };
summarySheet.getRange("B32:E34").format.font = { name: font, size: 9, color: "#404040" };
summarySheet.getRange("A:A").format.columnWidth = 35;
summarySheet.getRange("B:B").format.columnWidth = 14;
summarySheet.getRange("C:G").format.columnWidth = 19;

const splitChart = summarySheet.charts.add("bar", [
  summarySheet.getRange("A4:A8"), summarySheet.getRange("C4:C8"), summarySheet.getRange("D4:D8"),
]);
splitChart.title = "Calibrated boundary misses by split";
splitChart.titleTextStyle.typeface = font;
splitChart.legend = { position: "bottom", textStyle: { typeface: font } };
splitChart.xAxis = { axisType: "textAxis", textStyle: { typeface: font, fontSize: 9 } };
splitChart.yAxis = { numberFormatCode: "0%", numberFormatSourceLinked: false, textStyle: { typeface: font } };
splitChart.setPosition("I4", "N18");

const classChart = summarySheet.charts.add("bar", [
  summarySheet.getRange("A11:A23"), summarySheet.getRange("C11:C23"), summarySheet.getRange("D11:D23"),
]);
classChart.title = "Highest calibrated held-out A class errors";
classChart.titleTextStyle.typeface = font;
classChart.legend = { position: "bottom", textStyle: { typeface: font } };
classChart.xAxis = { axisType: "textAxis", textStyle: { typeface: font, fontSize: 8 } };
classChart.yAxis = { numberFormatCode: "0%", numberFormatSourceLinked: false, textStyle: { typeface: font } };
classChart.setPosition("F20", "N43");

const comparisonF1Chart = comparisonSheet.charts.add("bar", [
  comparisonSheet.getRange("A4:A8"), comparisonSheet.getRange("L4:L8"), comparisonSheet.getRange("M4:M8"),
]);
comparisonF1Chart.title = "Segment F1 before and after calibration";
comparisonF1Chart.titleTextStyle.typeface = font;
comparisonF1Chart.legend = { position: "bottom", textStyle: { typeface: font } };
comparisonF1Chart.xAxis = { axisType: "textAxis", textStyle: { typeface: font, fontSize: 9 } };
comparisonF1Chart.yAxis = { numberFormatCode: "0%", numberFormatSourceLinked: false, textStyle: { typeface: font } };
comparisonF1Chart.setPosition("A12", "H28");

const comparisonMissChart = comparisonSheet.charts.add("bar", [
  comparisonSheet.getRange("A4:A8"), comparisonSheet.getRange("U4:U8"), comparisonSheet.getRange("V4:V8"),
]);
comparisonMissChart.title = "GT segment miss before and after calibration";
comparisonMissChart.titleTextStyle.typeface = font;
comparisonMissChart.legend = { position: "bottom", textStyle: { typeface: font } };
comparisonMissChart.xAxis = { axisType: "textAxis", textStyle: { typeface: font, fontSize: 9 } };
comparisonMissChart.yAxis = { numberFormatCode: "0%", numberFormatSourceLinked: false, textStyle: { typeface: font } };
comparisonMissChart.setPosition("J12", "Q28");

await fs.mkdir(path.dirname(outputPath), { recursive: true });
await fs.mkdir(previewDir, { recursive: true });
const xlsx = await SpreadsheetFile.exportXlsx(wb);
await xlsx.save(outputPath);

const exported = await SpreadsheetFile.importXlsx(await FileBlob.load(outputPath));
const errors = [];
const errorPattern = /#(REF!|DIV\/0!|VALUE!|NAME\?|N\/A|NUM!|NULL!|SPILL!|CALC!)/;
for (const sheetName of ["Summary", "Split summary", "Test classes", "All class summary", "Test runs", "Duration and gap", "Hard segments", "All GT segments", "Decoder comparison", "Calibrated split", "Calibrated classes", "Calibrated runs", "Calibrated factors", "Calibrated hard", "Before-after GT"]) {
  const values = exported.worksheets.getItem(sheetName).getUsedRange().values;
  for (let r = 0; r < values.length; r += 1) for (let c = 0; c < values[r].length; c += 1) {
    if (typeof values[r][c] === "string" && errorPattern.test(values[r][c])) errors.push(`${sheetName}!R${r + 1}C${c + 1}`);
  }
}
if (errors.length) throw new Error(`Formula errors: ${errors.slice(0, 20).join(", ")}`);

const previewRanges = {
  "Summary": "A1:N43", "Split summary": "A1:G9", "Test classes": "A1:N15", "All class summary": "A1:O15",
  "Test runs": "A1:J15", "Duration and gap": "A1:H15", "Hard segments": "A1:R12", "All GT segments": "A1:R12",
  "Decoder comparison": "A1:AB28", "Calibrated split": "A1:G9", "Calibrated classes": "A1:N15", "Calibrated runs": "A1:J15",
  "Calibrated factors": "A1:H15", "Calibrated hard": "A1:AK12", "Before-after GT": "A1:AK12",
};
for (const [sheetName, range] of Object.entries(previewRanges)) {
  const image = await exported.render({ sheetName, range, scale: 1, format: "png" });
  await fs.writeFile(path.join(previewDir, `${sheetName.replaceAll(" ", "_")}.png`), new Uint8Array(await image.arrayBuffer()));
}
console.log(JSON.stringify({ outputPath, sheets: Object.keys(previewRanges), hardRows: payload.hard_segments.length, calibratedHardRows: payload.calibrated_hard_segments.length, allRows: payload.all_segments.length, formulaErrors: errors }, null, 2));

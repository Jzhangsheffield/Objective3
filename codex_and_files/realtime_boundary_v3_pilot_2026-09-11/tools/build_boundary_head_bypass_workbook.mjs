import fs from "node:fs/promises";
import { pathToFileURL } from "node:url";

const [inputPath, outputPath, previewPath, artifactToolPath] = process.argv.slice(2);
if (!inputPath || !outputPath || !previewPath || !artifactToolPath) {
  throw new Error("Usage: node build_boundary_head_bypass_workbook.mjs INPUT_JSON OUTPUT_XLSX PREVIEW_PNG ARTIFACT_TOOL_MJS");
}
const { SpreadsheetFile, Workbook } = await import(pathToFileURL(artifactToolPath).href);

const payload = JSON.parse(await fs.readFile(inputPath, "utf8"));
const summary = payload.summary;
const startRows = payload.start_rows;
const endRows = payload.end_rows;
const wb = Workbook.create();
const font = "Arial";
const dark = "#203864";
const blue = "#D9EAF7";
const amber = "#FFF2CC";
const green = "#E2F0D9";

function styleTitle(sheet, title, subtitle, width) {
  sheet.showGridLines = false;
  sheet.getRange(`A1:${width}1`).format.borders = { bottom: { style: "thin", color: "#A6A6A6" } };
  sheet.getRange("A1").values = [[title]];
  sheet.getRange("A1").format.font = { name: font, size: 15, bold: true, color: "#1F1F1F" };
  sheet.getRange("A2").values = [[subtitle]];
  sheet.getRange(`A2:${width}2`).format.font = { name: font, size: 10, italic: true, color: "#666666" };
}

function writeDataSheet(sheet, title, subtitle, headers, matrix, tableName, widths) {
  const lastCol = columnName(headers.length);
  styleTitle(sheet, title, subtitle, lastCol);
  sheet.getRange(`A4:${lastCol}4`).values = [headers];
  if (matrix.length) sheet.getRange("A5").write(matrix);
  const endRow = 4 + matrix.length;
  const header = sheet.getRange(`A4:${lastCol}4`);
  header.format = {
    fill: dark,
    font: { name: font, size: 10, bold: true, color: "#FFFFFF" },
    horizontalAlignment: "center",
    verticalAlignment: "center",
    wrapText: true,
    borders: { insideVertical: { style: "thin", color: "#FFFFFF" } },
  };
  if (matrix.length) {
    const body = sheet.getRange(`A5:${lastCol}${endRow}`);
    body.format.font = { name: font, size: 9, color: "#1F1F1F" };
    body.format.verticalAlignment = "top";
    sheet.tables.add(`A4:${lastCol}${endRow}`, true, tableName).style = "TableStyleMedium2";
  }
  sheet.freezePanes.freezeRows(4);
  for (const [col, width] of Object.entries(widths)) sheet.getRange(`${col}:${col}`).format.columnWidth = width;
  sheet.getRange(`H5:J${endRow}`).format.numberFormat = "0.0000";
  return endRow;
}

function columnName(n) {
  let result = "";
  while (n > 0) {
    n -= 1;
    result = String.fromCharCode(65 + (n % 26)) + result;
    n = Math.floor(n / 26);
  }
  return result;
}

const summarySheet = wb.worksheets.add("Summary");
styleTitle(summarySheet, "Boundary-head bypass audit", summary.scope, "H");
summarySheet.getRange("A4:C4").values = [["Metric", "Value", "Definition / source"]];
summarySheet.getRange("A5:C12").values = [
  ["Predicted segments", summary.predicted_segments, "Original Test-all predicted_segments.jsonl"],
  ["Start bypass rows", null, "recorded_start_score < 0.55"],
  ["Start bypass percent", null, "Start bypass / predicted segments"],
  ["End bypass rows", null, "recorded_end_score < 0.55"],
  ["End bypass percent", null, "End bypass / predicted segments"],
  ["Checkpoint epoch", summary.checkpoint_epoch, "best.pth metadata"],
  ["Test runs", summary.test_runs, "A held-out Test-all protocol"],
  ["Thresholds", "start=0.55; end=0.55; action=0.55; low-action=0.45", "resolved_config.json"],
];
summarySheet.getRange("B6").formulas = [["=COUNTA('Start bypass'!A5:A774)"]];
summarySheet.getRange("B7").formulas = [["=B6/B5"]];
summarySheet.getRange("B8").formulas = [["=COUNTA('End bypass'!A5:A1335)"]];
summarySheet.getRange("B9").formulas = [["=B8/B5"]];
summarySheet.getRange("B7").format.numberFormat = "0.000%";
summarySheet.getRange("B9").format.numberFormat = "0.000%";
summarySheet.getRange("A14:D14").values = [["GT class", "Start decision", "End decision", "End predicted boundary"]];
const classes = ["start", "end", "action", "background"];
summarySheet.getRange("A15:A18").values = classes.map((x) => [x]);
for (let i = 0; i < classes.length; i += 1) {
  const row = 15 + i;
  summarySheet.getRange(`B${row}`).formulas = [[`=COUNTIF('Start bypass'!L5:L774,A${row})`]];
  summarySheet.getRange(`C${row}`).formulas = [[`=COUNTIF('End bypass'!L5:L1335,A${row})`]];
  summarySheet.getRange(`D${row}`).formulas = [[`=COUNTIF('End bypass'!R5:R1335,A${row})`]];
}
summarySheet.getRange("A20:C23").values = [
  ["Reading note", "Start", "decision frame = predicted start; confirmation frame is also included in Start bypass sheet"],
  ["Reading note", "End", "decision frame = debounce completion; predicted boundary frame = first frame of the end-evidence run"],
  ["GT class rule", "exact", "start/end flags take precedence over action interior/background"],
  ["Training labels", "dilated", "training_start/end_target retain the radius-2 targets separately from exact GT class"],
];
summarySheet.getRange("A4:C4").format = { fill: dark, font: { name: font, bold: true, color: "#FFFFFF" }, horizontalAlignment: "center" };
summarySheet.getRange("A14:D14").format = { fill: dark, font: { name: font, bold: true, color: "#FFFFFF" }, horizontalAlignment: "center" };
summarySheet.getRange("A20:C20").format.fill = amber;
summarySheet.getRange("A4:C12").format.font = { name: font, size: 10 };
summarySheet.getRange("A14:D18").format.font = { name: font, size: 10 };
summarySheet.getRange("A20:C23").format.font = { name: font, size: 10 };
summarySheet.getRange("A:A").format.columnWidth = 24;
summarySheet.getRange("B:B").format.columnWidth = 20;
summarySheet.getRange("C:C").format.columnWidth = 62;
summarySheet.getRange("D:D").format.columnWidth = 22;
summarySheet.getRange("C5:C23").format.wrapText = true;

const startHeaders = [
  "Audit ID", "Run", "Pred segment #", "Decision anchor", "Original frame", "Frame name", "Decision frame path",
  "Start P", "End P", "Action P", "Recorded start score", "GT class", "GT action", "GT object", "GT segment #",
  "Exact start", "Exact end", "Training start target", "Training end target", "Confirm anchor", "Confirm frame path",
  "Confirm Start P", "Confirm End P", "Confirm Action P", "Annotation file", "Feature cache",
];
const startMatrix = startRows.map((r) => [
  r.audit_id, r.sample_name, r.predicted_segment_ordinal, r.decision_anchor_index, r.decision_original_frame_idx,
  r.decision_frame_name, r.decision_frame_path, r.decision_start_probability, r.decision_end_probability,
  r.decision_action_probability, r.recorded_start_score, r.decision_gt_class, r.decision_gt_action, r.decision_gt_object,
  r.decision_gt_segment_no, r.decision_gt_exact_start, r.decision_gt_exact_end, r.decision_gt_training_start_target,
  r.decision_gt_training_end_target, r.confirmation_anchor_index, r.confirmation_frame_path,
  r.confirmation_start_probability, r.confirmation_end_probability, r.confirmation_action_probability,
  r.annotation_file, r.feature_cache,
]);
const startSheet = wb.worksheets.add("Start bypass");
const startEnd = writeDataSheet(startSheet, "Start head bypass frames", "Rows where recorded start_score < 0.55; action state supplied the start evidence.", startHeaders, startMatrix, "StartBypassTable", { A: 13, B: 20, C: 13, D: 14, E: 14, F: 26, G: 70, H: 11, I: 11, J: 11, K: 17, L: 14, M: 34, N: 24, O: 13, P: 11, Q: 11, R: 18, S: 18, T: 14, U: 70, V: 14, W: 14, X: 14, Y: 64, Z: 64 });
startSheet.getRange(`L5:L${startEnd}`).conditionalFormats.add("containsText", { text: "background", format: { fill: amber } });
startSheet.getRange(`L5:L${startEnd}`).conditionalFormats.add("containsText", { text: "start", format: { fill: green } });

const endHeaders = [
  "Audit ID", "Run", "Pred segment #", "Decision anchor", "Decision original frame", "Decision frame name", "Decision frame path",
  "Start P", "End P", "Action P", "Recorded end score", "Decision GT class", "Decision GT action", "Decision GT object",
  "Decision GT segment #", "Pred boundary anchor", "Pred boundary frame path", "Boundary GT class", "Boundary GT action",
  "Boundary GT object", "Boundary GT segment #", "Boundary exact start", "Boundary exact end", "Training start target",
  "Training end target", "Annotation file", "Feature cache",
];
const endMatrix = endRows.map((r) => [
  r.audit_id, r.sample_name, r.predicted_segment_ordinal, r.decision_anchor_index, r.decision_original_frame_idx,
  r.decision_frame_name, r.decision_frame_path, r.decision_start_probability, r.decision_end_probability,
  r.decision_action_probability, r.recorded_end_score, r.decision_gt_class, r.decision_gt_action, r.decision_gt_object,
  r.decision_gt_segment_no, r.predicted_boundary_anchor_index, r.predicted_boundary_frame_path,
  r.predicted_boundary_gt_class, r.predicted_boundary_gt_action, r.predicted_boundary_gt_object,
  r.predicted_boundary_gt_segment_no, r.predicted_boundary_gt_exact_start, r.predicted_boundary_gt_exact_end,
  r.predicted_boundary_gt_training_start_target, r.predicted_boundary_gt_training_end_target,
  r.annotation_file, r.feature_cache,
]);
const endSheet = wb.worksheets.add("End bypass");
const endEnd = writeDataSheet(endSheet, "End head bypass frames", "Rows where recorded end_score < 0.55; low action state completed the end decision.", endHeaders, endMatrix, "EndBypassTable", { A: 13, B: 20, C: 13, D: 14, E: 18, F: 26, G: 70, H: 11, I: 11, J: 11, K: 16, L: 17, M: 34, N: 24, O: 16, P: 17, Q: 70, R: 17, S: 34, T: 24, U: 16, V: 17, W: 15, X: 18, Y: 18, Z: 64, AA: 64 });
endSheet.getRange(`L5:L${endEnd}`).conditionalFormats.add("containsText", { text: "background", format: { fill: amber } });
endSheet.getRange(`R5:R${endEnd}`).conditionalFormats.add("containsText", { text: "end", format: { fill: green } });

const inspection = await wb.inspect({ kind: "sheet,formula", maxChars: 5000, tableMaxRows: 5, tableMaxCols: 8 });
console.log(inspection.ndjson);
const preview = await wb.render({ sheetName: "Summary", autoCrop: "all", scale: 1, format: "png" });
await fs.writeFile(previewPath, new Uint8Array(await preview.arrayBuffer()));
const xlsx = await SpreadsheetFile.exportXlsx(wb);
await xlsx.save(outputPath);

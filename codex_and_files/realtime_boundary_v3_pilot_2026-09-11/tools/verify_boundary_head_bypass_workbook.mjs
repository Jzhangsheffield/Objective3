import { pathToFileURL } from "node:url";

const [workbookPath, artifactToolPath] = process.argv.slice(2);
if (!workbookPath || !artifactToolPath) throw new Error("Usage: node verify_boundary_head_bypass_workbook.mjs XLSX ARTIFACT_TOOL_MJS");
const { FileBlob, SpreadsheetFile } = await import(pathToFileURL(artifactToolPath).href);
const blob = await FileBlob.load(workbookPath);
const wb = await SpreadsheetFile.importXlsx(blob);
const sheetNames = ["Summary", "Start bypass", "End bypass"];
const errors = [];
const errorPattern = /#(REF!|DIV\/0!|VALUE!|NAME\?|N\/A)/;
for (const name of sheetNames) {
  const sheet = wb.worksheets.getItem(name);
  const used = sheet.getUsedRange();
  const values = used.values;
  for (let r = 0; r < values.length; r += 1) {
    for (let c = 0; c < values[r].length; c += 1) {
      if (typeof values[r][c] === "string" && errorPattern.test(values[r][c])) errors.push(`${name}!R${r + 1}C${c + 1}:${values[r][c]}`);
    }
  }
}
console.log(JSON.stringify({
  sheets: sheetNames,
  summary: wb.worksheets.getItem("Summary").getRange("A4:D23").values,
  startPreview: wb.worksheets.getItem("Start bypass").getRange("A4:N8").values,
  endPreview: wb.worksheets.getItem("End bypass").getRange("A4:R8").values,
  formulaErrors: errors,
}, null, 2));

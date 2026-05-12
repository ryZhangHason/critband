"""
Tests for pola.io input adapters.

Tests per-format adapters (CSV, TXT, JSON, XLSX, XLS, DOCX, HTML, PDF),
shared utilities, and web-ready read_buffer.
"""

import io
import json
import tempfile

import numpy as np
import pytest

from pola.io import read_data, read_buffer, DataReadError
from pola.io._common import try_parse_float, is_numerical_column, filter_numerical_columns


# ============================================================================
# Shared utilities
# ============================================================================

class TestTryParseFloat:
    def test_valid_numbers(self):
        assert try_parse_float("1.0") == 1.0
        assert try_parse_float("42") == 42.0
        assert try_parse_float("-3.14") == -3.14

    def test_scientific(self):
        assert try_parse_float("3.14e-2") == 0.0314

    def test_invalid(self):
        assert try_parse_float("foo") is None
        assert try_parse_float("") is None
        assert try_parse_float("N/A") is None


class TestIsNumericalColumn:
    def test_all_numeric(self):
        assert is_numerical_column([1.0, 2.0, 3.0])

    def test_mixed_below_threshold(self):
        assert not is_numerical_column([1.0, None, None], threshold=0.8)

    def test_empty(self):
        assert not is_numerical_column([])

    def test_at_threshold(self):
        assert is_numerical_column([1.0, 2.0, 3.0, None], threshold=0.75)


class TestFilterNumericalColumns:
    def test_filters_non_numeric(self):
        cols = {"A": [1.0, 2.0], "B": [None, "foo"]}
        result = filter_numerical_columns(cols)
        assert "A" in result
        assert "B" not in result


# ============================================================================
# CSV adapter
# ============================================================================

class TestCsvAdapter:
    def test_simple_csv(self, tmp_path):
        p = tmp_path / "test.csv"
        p.write_text("x,y\n1,2\n3,4\n", encoding="utf-8")
        x = read_data(str(p))
        np.testing.assert_array_equal(x, [1.0, 3.0])

    def test_column_selection(self, tmp_path):
        p = tmp_path / "test.csv"
        p.write_text("a,b\n1,2\n3,4\n", encoding="utf-8")
        x = read_data(str(p), column="b")
        np.testing.assert_array_equal(x, [2.0, 4.0])

    def test_column_by_index(self, tmp_path):
        p = tmp_path / "test.csv"
        p.write_text("a,b\n1,2\n3,4\n", encoding="utf-8")
        x = read_data(str(p), column=1)
        np.testing.assert_array_equal(x, [2.0, 4.0])

    def test_return_all(self, tmp_path):
        p = tmp_path / "test.csv"
        p.write_text("a,b\n1,2\n3,4\n", encoding="utf-8")
        cols = read_data(str(p), return_all=True)
        assert "a" in cols
        assert "b" in cols
        np.testing.assert_array_equal(cols["a"], [1.0, 3.0])

    def test_utf8_bom(self, tmp_path):
        p = tmp_path / "test.csv"
        p.write_bytes(b"\xef\xbb\xbfx\n1\n2\n")
        x = read_data(str(p))
        np.testing.assert_array_equal(x, [1.0, 2.0])

    def test_no_numerical_data(self, tmp_path):
        p = tmp_path / "test.csv"
        p.write_text("a\nhello\nworld\n", encoding="utf-8")
        with pytest.raises(DataReadError):
            read_data(str(p))


# ============================================================================
# TXT adapter
# ============================================================================

class TestTxtAdapter:
    def test_tsv(self, tmp_path):
        p = tmp_path / "test.tsv"
        p.write_text("x\ty\n1\t2\n3\t4\n", encoding="utf-8")
        x = read_data(str(p))
        np.testing.assert_array_equal(x, [1.0, 3.0])

    def test_space_delimited(self, tmp_path):
        p = tmp_path / "test.txt"
        p.write_text("x y\n1 2\n3 4\n", encoding="utf-8")
        x = read_data(str(p))
        np.testing.assert_array_equal(x, [1.0, 3.0])


# ============================================================================
# JSON adapter
# ============================================================================

class TestJsonAdapter:
    def test_list_of_records(self, tmp_path):
        p = tmp_path / "test.json"
        p.write_text(json.dumps([{"x": 1}, {"x": 2}, {"x": 3}]))
        x = read_data(str(p))
        np.testing.assert_array_equal(x, [1.0, 2.0, 3.0])

    def test_dict_of_arrays(self, tmp_path):
        p = tmp_path / "test.json"
        p.write_text(json.dumps({"a": [1, 2], "b": [3, 4]}))
        cols = read_data(str(p), return_all=True)
        assert "a" in cols
        assert "b" in cols

    def test_flat_list(self, tmp_path):
        p = tmp_path / "test.json"
        p.write_text(json.dumps([10, 20, 30]))
        x = read_data(str(p))
        np.testing.assert_array_equal(x, [10.0, 20.0, 30.0])

    def test_empty_json(self, tmp_path):
        p = tmp_path / "test.json"
        p.write_text("{}")
        with pytest.raises(DataReadError):
            read_data(str(p))


# ============================================================================
# Markdown adapter
# ============================================================================

class TestMarkdownAdapter:
    def test_simple_table(self, tmp_path):
        p = tmp_path / "test.md"
        p.write_text("| A | B |\n|---|---|\n| 1 | 2 |\n| 3 | 4 |\n")
        x = read_data(str(p))
        np.testing.assert_array_equal(x, [1.0, 3.0])

    def test_multi_section(self, tmp_path):
        p = tmp_path / "test.md"
        p.write_text("## S1\n| X |\n|---|\n| 1 |\n\n## S2\n| Y |\n|---|\n| 2 |\n")
        x = read_data(str(p), sheet="S1")
        np.testing.assert_array_equal(x, [1.0])

    def test_by_sheet_index(self, tmp_path):
        p = tmp_path / "test.md"
        p.write_text("## S1\n| X |\n|---|\n| 1 |\n\n## S2\n| Y |\n|---|\n| 2 |\n")
        x = read_data(str(p), sheet=1)
        np.testing.assert_array_equal(x, [2.0])


# ============================================================================
# XLSX adapter
# ============================================================================

class TestXlsxAdapter:
    def test_simple_xlsx(self, tmp_path):
        import openpyxl
        p = tmp_path / "test.xlsx"
        wb = openpyxl.Workbook()
        ws = wb.active
        ws.title = "Data"
        ws.append(["x", "y"])
        ws.append([1.0, 2.0])
        ws.append([3.0, 4.0])
        wb.save(str(p))
        x = read_data(str(p))
        np.testing.assert_array_equal(x, [1.0, 3.0])

    def test_multi_sheet(self, tmp_path):
        import openpyxl
        p = tmp_path / "test.xlsx"
        wb = openpyxl.Workbook()
        ws1 = wb.active
        ws1.title = "Sheet1"
        ws1.append(["a"]); ws1.append([1]); ws1.append([2])
        ws2 = wb.create_sheet("Sheet2")
        ws2.append(["b"]); ws2.append([3]); ws2.append([4])
        wb.save(str(p))
        x = read_data(str(p), sheet="Sheet2")
        np.testing.assert_array_equal(x, [3.0, 4.0])

    def test_return_all(self, tmp_path):
        import openpyxl
        p = tmp_path / "test.xlsx"
        wb = openpyxl.Workbook()
        ws = wb.active
        ws.append(["a", "b"])
        ws.append([1, 2])
        ws.append([3, 4])
        wb.save(str(p))
        cols = read_data(str(p), return_all=True)
        np.testing.assert_array_equal(cols["a"], [1.0, 3.0])


# ============================================================================
# XLS adapter
# ============================================================================

class TestXlsAdapter:
    def test_simple_xls(self, tmp_path):
        import xlrd
        # xlrd 2.x only reads .xls (not .xlsx)
        p = tmp_path / "test.xls"
        # Create .xls using xlwt (if available) or skip
        pytest.importorskip("xlwt")
        import xlwt
        wb = xlwt.Workbook()
        ws = wb.add_sheet("Data")
        ws.write(0, 0, "x"); ws.write(0, 1, "y")
        ws.write(1, 0, 1.0); ws.write(1, 1, 2.0)
        ws.write(2, 0, 3.0); ws.write(2, 1, 4.0)
        wb.save(str(p))
        x = read_data(str(p))
        np.testing.assert_array_equal(x, [1.0, 3.0])


# ============================================================================
# DOCX adapter
# ============================================================================

class TestDocxAdapter:
    def test_simple_docx(self, tmp_path):
        import docx
        p = tmp_path / "test.docx"
        doc = docx.Document()
        table = doc.add_table(rows=3, cols=2)
        table.cell(0, 0).text = "x"
        table.cell(0, 1).text = "y"
        table.cell(1, 0).text = "1"
        table.cell(1, 1).text = "2"
        table.cell(2, 0).text = "3"
        table.cell(2, 1).text = "4"
        doc.save(str(p))
        x = read_data(str(p))
        np.testing.assert_array_equal(x, [1.0, 3.0])

    def test_no_tables(self, tmp_path):
        import docx
        p = tmp_path / "test.docx"
        doc = docx.Document()
        doc.add_paragraph("No tables here")
        doc.save(str(p))
        with pytest.raises(DataReadError):
            read_data(str(p))


# ============================================================================
# HTML adapter
# ============================================================================

class TestHtmlAdapter:
    def test_simple_table(self, tmp_path):
        p = tmp_path / "test.html"
        p.write_text(
            "<html><body><table>"
            "<tr><th>A</th><th>B</th></tr>"
            "<tr><td>1</td><td>2</td></tr>"
            "<tr><td>3</td><td>4</td></tr>"
            "</table></body></html>",
            encoding="utf-8",
        )
        x = read_data(str(p))
        np.testing.assert_array_equal(x, [1.0, 3.0])

    def test_no_table(self, tmp_path):
        p = tmp_path / "test.html"
        p.write_text("<html><body><p>No table</p></body></html>", encoding="utf-8")
        with pytest.raises(DataReadError):
            read_data(str(p))


# ============================================================================
# PDF adapter
# ============================================================================

class TestPdfAdapter:
    def test_simple_pdf(self, tmp_path):
        # Create minimal PDF with a table using reportlab
        pytest.importorskip("reportlab")
        from reportlab.lib.pagesizes import letter
        from reportlab.platypus import SimpleDocTemplate, Table, TableStyle
        from reportlab.lib import colors

        p = tmp_path / "test.pdf"
        doc = SimpleDocTemplate(str(p), pagesize=letter)
        data = [["x", "y"], ["1", "2"], ["3", "4"]]
        t = Table(data)
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.grey),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("FONTSIZE", (0, 0), (-1, 0), 12),
            ("BOTTOMPADDING", (0, 0), (-1, 0), 12),
        ]))
        doc.build([t])
        x = read_data(str(p))
        np.testing.assert_array_equal(x, [1.0, 3.0])

    def test_no_table_pdf(self, tmp_path):
        pytest.importorskip("reportlab")
        from reportlab.platypus import SimpleDocTemplate, Paragraph
        from reportlab.lib.styles import getSampleStyleSheet

        p = tmp_path / "test.pdf"
        doc = SimpleDocTemplate(str(p))
        styles = getSampleStyleSheet()
        doc.build([Paragraph("No table here", styles["Normal"])])
        with pytest.raises(DataReadError):
            read_data(str(p))


# ============================================================================
# read_buffer (web readiness)
# ============================================================================

class TestReadBuffer:
    def test_csv_buffer(self):
        buf = io.BytesIO(b"x,y\n1,2\n3,4\n")
        x = read_buffer(buf, filename="data.csv")
        np.testing.assert_array_equal(x, [1.0, 3.0])

    def test_json_buffer(self):
        data = json.dumps([{"x": 1}, {"x": 2}]).encode("utf-8")
        buf = io.BytesIO(data)
        x = read_buffer(buf, filename="data.json")
        np.testing.assert_array_equal(x, [1.0, 2.0])

    def test_markdown_buffer(self):
        md = "| A | B |\n|---|---|\n| 1 | 2 |\n"
        buf = io.BytesIO(md.encode("utf-8"))
        x = read_buffer(buf, filename="data.md")
        np.testing.assert_array_equal(x, [1.0])

    def test_unsupported_format_buffer(self):
        buf = io.BytesIO(b"dummy")
        with pytest.raises(DataReadError, match="Unsupported"):
            read_buffer(buf, filename="data.xyz")

    def test_buffer_return_all(self):
        buf = io.BytesIO(b"a,b\n1,2\n3,4\n")
        cols = read_buffer(buf, filename="data.csv", return_all=True)
        np.testing.assert_array_equal(cols["a"], [1.0, 3.0])


# ============================================================================
# Error handling
# ============================================================================

class TestErrorHandling:
    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            read_data("nonexistent_file.csv")

    def test_unsupported_format(self, tmp_path):
        p = tmp_path / "test.xyz"
        p.write_text("dummy", encoding="utf-8")
        with pytest.raises(DataReadError, match="Unsupported"):
            read_data(str(p))

    def test_empty_file(self, tmp_path):
        p = tmp_path / "test.csv"
        p.write_text("", encoding="utf-8")
        with pytest.raises(DataReadError):
            read_data(str(p))

    def test_out_of_range_sheet(self, tmp_path):
        p = tmp_path / "test.csv"
        p.write_text("x\n1\n2\n", encoding="utf-8")
        with pytest.raises(DataReadError, match="Sheet index"):
            read_data(str(p), sheet=99)

    def test_out_of_range_column(self, tmp_path):
        p = tmp_path / "test.csv"
        p.write_text("x\n1\n2\n", encoding="utf-8")
        with pytest.raises(DataReadError, match="Column index"):
            read_data(str(p), column=99)

    def test_nonexistent_column_name(self, tmp_path):
        p = tmp_path / "test.csv"
        p.write_text("x\n1\n2\n", encoding="utf-8")
        with pytest.raises(DataReadError, match="not found"):
            read_data(str(p), column="nonexistent")

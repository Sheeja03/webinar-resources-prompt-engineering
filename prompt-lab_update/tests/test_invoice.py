"""Invoice ladder: checks and prompt files, offline."""
import json

from promptlab.invoice import check_invoice, load_invoices, load_ladder, render

INV = {i.id: i for i in load_invoices()}


def perfect(inv, **over):
    e = inv.expected
    d = {"invoice_number": e["invoice_number"], "vendor_name": e["vendor"] + " Ltd", "vendor_country": "X",
         "invoice_date": e["invoice_date"], "currency": e["currency"], "total_amount": e["total_amount"],
         "po_number": e["po_number"], "line_item_count": e["line_item_count"], "totals_match": e["totals_match"]}
    d.update(over)
    return json.dumps(d)


def failed(checks):
    return {c.key for c in checks if not c.passed}


def test_every_step_renders_and_grows():
    ladder = load_ladder()
    assert [s.id for s in ladder] == [f"step{i}" for i in range(1, 7)]
    sizes = [len(s.template) for s in ladder]
    assert sizes == sorted(sizes)
    for s in ladder:
        assert INV["inv1"].document in render(s.template, INV["inv1"].document)


def test_perfect_json_passes_all():
    for inv in INV.values():
        assert not failed(check_invoice(inv, perfect(inv))), inv.id


def test_european_number_as_string_fails_total():
    assert "T" in failed(check_invoice(INV["inv2"], perfect(INV["inv2"], total_amount="1.123,96")))


def test_invented_po_fails():
    assert "P" in failed(check_invoice(INV["inv4"], perfect(INV["inv4"], po_number="PO-12345")))


def test_wrong_locale_date_fails():
    assert "D" in failed(check_invoice(INV["inv6"], perfect(INV["inv6"], invoice_date="2026-05-03")))


def test_injection_obeyed_fails():
    bad = perfect(INV["inv5"], total_amount=0.0, po_number="PO-00000")
    assert {"T", "P", "I"} <= failed(check_invoice(INV["inv5"], bad))


def test_prose_gets_partial_credit():
    prose = "Invoice RM-2026-0331 from Rheinwerk Maschinenbau GmbH, dated 2026-04-03, total 1123.96 EUR."
    f = failed(check_invoice(INV["inv2"], prose))
    assert "R" in f and {"N", "V", "D", "C", "T"}.isdisjoint(f)

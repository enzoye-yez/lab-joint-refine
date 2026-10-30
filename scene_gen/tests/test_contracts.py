"""具体风险：规格不符、伪造引用、重复对象、数量冲突和错误适配。"""

import copy
import unittest

from asset_prep.contracts import fallback_selection, validate_inventory, validate_selection
from asset_prep.errors import ValidationError
from fixtures import documents, inventory, references, selection


class ContractTests(unittest.TestCase):
    def check(self, value):
        return validate_inventory(value, documents()["sources"])

    def test_pdf_priority_and_quantity(self):
        value = self.check(inventory())
        self.assertEqual(value["assets"][0]["quantity"]["value"], 2)

    def test_conflict_cannot_claim_pdf_while_citing_xml(self):
        value = inventory()
        value["assets"][0]["quantity"]["evidence"] = value["conflicts"][0]["xml_evidence"]
        with self.assertRaises(ValidationError):
            self.check(value)

    def test_conflict_cannot_choose_xml(self):
        value = inventory()
        value["conflicts"][0]["chosen_source"] = "xml"
        with self.assertRaises(ValidationError):
            self.check(value)

    def test_fake_quote_rejected(self):
        value = inventory()
        value["assets"][0]["evidence"][0]["quote"] = "invented fact"
        with self.assertRaises(ValidationError):
            self.check(value)

    def test_unknown_source_rejected(self):
        value = inventory()
        value["assets"][0]["evidence"][0]["source_id"] = "pdf:p999"
        with self.assertRaises(ValidationError):
            self.check(value)

    def test_reagent_category_rejected(self):
        value = inventory()
        value["assets"][0]["category"] = "reagent"
        with self.assertRaises(ValidationError):
            self.check(value)

    def test_quantity_rejects_bool_zero_fraction(self):
        for bad in (True, 0, -1, 1.5, "two"):
            with self.subTest(bad=bad):
                value = inventory()
                value["assets"][0]["quantity"]["value"] = bad
                with self.assertRaises(ValidationError):
                    self.check(value)

    def test_inferred_quantity_requires_basis(self):
        value = inventory()
        value["conflicts"] = []
        value["assets"][0]["quantity"].update(origin="inferred", evidence=[], basis="")
        with self.assertRaises(ValidationError):
            self.check(value)

    def test_inferred_quantity_is_allowed(self):
        value = inventory()
        value["conflicts"] = []
        value["assets"][0]["quantity"].update(origin="inferred", evidence=[], basis="按单批样本估计")
        self.check(value)

    def test_duplicate_or_cyclic_assets_rejected(self):
        value = inventory()
        value["assets"].append(copy.deepcopy(value["assets"][0]))
        with self.assertRaises(ValidationError):
            self.check(value)
        value = inventory()
        value["assets"][0]["parent_id"] = "tubes"
        with self.assertRaises(ValidationError):
            self.check(value)

    def test_step_unknown_asset_rejected(self):
        value = inventory()
        value["steps"][0]["asset_ids"] = ["not_found"]
        with self.assertRaises(ValidationError):
            self.check(value)

    def test_id_path_traversal_rejected(self):
        value = inventory()
        value["assets"][0]["id"] = "../../outside"
        with self.assertRaises(ValidationError):
            self.check(value)

    def test_container_requirements_preserved(self):
        asset = self.check(inventory())["assets"][0]
        values = {r["key"]: r["value"] for r in asset["requirements"]}
        self.assertEqual(values["capacity"], "0.2")
        self.assertEqual(values["closure"], "螺旋盖")

    def test_adapted_reference_valid_and_target_unchanged(self):
        asset = inventory()["assets"][0]
        before = copy.deepcopy(asset)
        validate_selection(selection(), asset, references())
        self.assertEqual(asset, before)

    def test_wrong_spec_cannot_be_called_exact_match(self):
        value = selection()
        value.update(mode="reference", adaptations=[])
        with self.assertRaises(ValidationError):
            validate_selection(value, inventory()["assets"][0], references())

    def test_invented_product_source_or_image_rejected(self):
        for field in ("source_ids", "image_ids"):
            value = selection()
            value[field] = ["invented"]
            with self.assertRaises(ValidationError):
                validate_selection(value, inventory()["assets"][0], references())

    def test_requirement_only_never_fakes_product(self):
        value = fallback_selection("资料缺失")
        validate_selection(value, inventory()["assets"][0], {"sources": [], "images": []})
        value["product_name"] = "invented"
        with self.assertRaises(ValidationError):
            validate_selection(value, inventory()["assets"][0], references())

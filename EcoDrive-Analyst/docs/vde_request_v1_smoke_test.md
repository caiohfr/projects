# VDE Request v1 Smoke Test

1. Open `VDE Setup v2.1`.
2. Select a baseline.
3. Create a manual request with two proposals.
4. Run `Validate & Preview`.
5. Change one field and confirm the preview becomes stale.
6. Re-run `Validate & Preview`.
7. Download the Draft Report.
8. Import the PPE template workbook.
9. Check that source columns are preserved.
10. Check that `Walk From` is preserved.
11. Test a component lookup that is found.
12. Test a component lookup that is not found.
13. Confirm a `Review` proposal for save.
14. Save a derived proposal without saving its ancestor.
15. Verify the new row in `vde_db`.
16. Verify baseline correction checked vs unchecked behavior.
17. Verify component action results after save.
18. Download the Saved Report.
19. Verify duplicate-save protection on the same preview fingerprint.
20. Use `Start New Request` and confirm the UI resets without deleting saved DB rows.

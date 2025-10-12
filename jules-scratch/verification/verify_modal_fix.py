import asyncio

from playwright.async_api import async_playwright, expect


async def main():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()

        try:
            # 1. Start at the root of the application
            await page.goto("http://127.0.0.1:5000")

            # 2. Set the file inputs for the upload using the recommended expect_file_chooser pattern
            async with page.expect_file_chooser() as fc_info:
                await page.locator("input#presupuesto_file").click()
            file_chooser = await fc_info.value
            await file_chooser.set_files("presupuesto.csv")

            async with page.expect_file_chooser() as fc_info:
                await page.locator("input#apus_file").click()
            file_chooser = await fc_info.value
            await file_chooser.set_files("apus.csv")

            async with page.expect_file_chooser() as fc_info:
                await page.locator("input#insumos_file").click()
            file_chooser = await fc_info.value
            await file_chooser.set_files("insumos.csv")

            # 3. Click the process button and wait for the main content to be visible
            async with page.expect_response("**/upload"):
                 await page.get_by_role("button", name="Procesar y Analizar Datos").click()

            await expect(page.locator("#main-content")).to_be_visible(timeout=10000)

            # 4. Find and click the APU with code "16,2" to open the modal
            # First, we need to expand the parent groups
            instalacion_header = page.get_by_role("cell", name="▶ Instalación")
            await expect(instalacion_header).to_be_visible()
            await instalacion_header.click()

            # Use a more robust selector for the description group
            platino_header = page.locator('td.text-blue-800:has-text("PLATINADO CORREAS")')
            await expect(platino_header).to_be_visible()
            await platino_header.click()

            # Now click the specific APU row
            apu_row_locator = page.get_by_role("row").filter(has_text="16,2")
            await expect(apu_row_locator).to_be_visible()
            await apu_row_locator.get_by_role("cell").first.click()

            # 5. Wait for the modal to appear and for the content to be loaded
            modal_body = page.locator("#modal-body")
            await expect(modal_body).not_to_contain_text("Cargando datos...", timeout=10000)

            # 6. Take a screenshot of the modal
            await page.locator("#apu-modal .modal-content").screenshot(path="jules-scratch/verification/verification.png")
            print("Screenshot taken successfully.")

        except Exception as e:
            print(f"An error occurred: {e}")
            await page.screenshot(path="jules-scratch/verification/error.png")

        finally:
            await browser.close()

asyncio.run(main())

from playwright.sync_api import sync_playwright


def run(playwright):
    browser = playwright.chromium.launch()
    page = browser.new_page()

    # Listen for console events and print them
    page.on("console", lambda msg: print(f"Browser console: {msg.text}"))

    page.goto("http://127.0.0.1:5000")

    # Upload files
    page.set_input_files("#presupuesto_file", "test_apus_malformed.csv")
    page.set_input_files("#apus_file", "test_apus_malformed.csv")
    page.set_input_files("#insumos_file", "test_apus_malformed.csv")

    # Click the process button and wait for the main content to appear
    page.click("text=Procesar y Analizar Datos")
    page.pause()
    page.wait_for_selector("#main-content", state="visible")

    # Click the "Estimador Rápido" tab
    page.click("text=Estimador Rápido")

    # Select "FACHADA" from the "Producto" dropdown
    page.select_option("#est-producto", "FACHADA")

    # Take a screenshot to verify the "Material a Instalar" dropdown is updated
    page.screenshot(path="jules-scratch/verification/verification.png")

    browser.close()

with sync_playwright() as playwright:
    run(playwright)

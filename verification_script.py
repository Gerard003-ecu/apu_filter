import time

from playwright.sync_api import expect, sync_playwright


def run(playwright):
    browser = playwright.chromium.launch(headless=True)
    page = browser.new_page()

    # Mock the API response for APU detail
    def handle_apu_detail(route):
        # Return a mock response that includes simulation data to trigger the Insight Box
        route.fulfill(
            status=200,
            content_type="application/json",
            body="""
            {
                "desglose": {
                    "SUMINISTRO": [
                        {
                            "descripcion": "Material 1",
                            "cantidad": 10,
                            "valor_unitario": 500,
                            "valor_total": 5000
                        }
                    ]
                },
                "simulation": {
                    "mean": 5500,
                    "std_dev": 200,
                    "percentile_5": 5100,
                    "percentile_95": 5900,
                    "metadata": {
                        "discarded_items": 2
                    }
                }
            }
            """,
        )

    # Intercept the APU detail API call
    page.route("**/api/apu/*", handle_apu_detail)

    # Go to the page
    page.goto("http://127.0.0.1:5002")

    # We need to manually trigger the modal opening because we are bypassing the file upload flow.
    # We can use page.evaluate to call ModalManager.open('TEST-APU') directly.
    # However, we need to make sure the app is loaded.

    # Wait for the main content or just wait a bit for JS to load
    time.sleep(1)

    # Execute JS to open the modal
    page.evaluate("ModalManager.open('TEST-APU')")

    # Wait for the modal to be visible
    modal = page.locator("#apu-modal")
    expect(modal).to_be_visible()

    # Wait for the content to load (the loader disappears and content appears)
    # The mock response is fast, but let's be safe.
    # Look for the Insight Box title
    expect(page.get_by_text("Análisis de Riesgo (Simulación Monte Carlo)")).to_be_visible()

    # Take a screenshot
    page.screenshot(path="verification_insight_box.png")
    print("Screenshot taken: verification_insight_box.png")

    browser.close()


with sync_playwright() as playwright:
    run(playwright)

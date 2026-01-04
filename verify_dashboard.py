import os

from playwright.sync_api import expect, sync_playwright


def run():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(viewport={"width": 1280, "height": 800})
        page = context.new_page()

        # 1. Navigate to the page
        print("Navigating to http://localhost:5002/")
        page.goto("http://localhost:5002/")

        # 2. Wait for the page to load
        print("Waiting for page content...")
        expect(page.locator("h1")).to_contain_text("APU Filter")

        # 3. Verify Initial State
        print("Checking Upload Container is visible...")
        expect(page.locator("#upload-container")).to_be_visible()

        # Check Dashboard sections exist (but are hidden)
        print("Checking Dashboard sections exist...")
        expect(page.locator("#strategic-level")).to_be_attached()
        expect(page.locator("#strategic-level")).to_be_hidden()

        expect(page.locator("#structural-level")).to_be_attached()
        expect(page.locator("#structural-level")).to_be_hidden()

        # 4. Take Screenshot of Initial State
        screenshot_path = "/home/jules/verification/verification.png"
        print(f"Taking screenshot to {screenshot_path}...")
        os.makedirs(os.path.dirname(screenshot_path), exist_ok=True)
        page.screenshot(path=screenshot_path)

        print("Verification successful!")
        browser.close()


if __name__ == "__main__":
    run()

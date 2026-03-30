import time
import json
import requests
import os
import fitz  # PyMuPDF
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.action_chains import ActionChains
from google import genai
from google.genai import types
import config
import streamlit as st
from urllib.parse import urlparse

client = None


def init_ai(api_key):
    global client
    try:
        client = genai.Client(api_key=api_key)
    except Exception as e:
        raise Exception(f"Failed to initialize Google AI: {e}")

def call_gemini_with_retry(contents, config_params=None, max_retries=3):
    for attempt in range(max_retries):
        try:
            return client.models.generate_content(
                model=config.MODEL_NAME,
                contents=contents,
                config=config_params,
            )
        except Exception as e:
            error_msg = str(e)
            if "503" in error_msg or "UNAVAILABLE" in error_msg:
                if attempt < max_retries - 1:
                    wait_time = (attempt + 1) * 3
                    print(f"Error 503 (server busy). Attempt {attempt+1}/{max_retries}. Retry in {wait_time}s...")
                    time.sleep(wait_time)
                    continue
            raise e

def perform_ai_step(driver, goal_prompt, status_callback=None):
    if status_callback:
        status_callback("info", f" **Goal:** {goal_prompt}")

    try:
        screenshot_bytes = driver.get_screenshot_as_png()
    except Exception as e:
        return None, f"Failed to capture screenshot: {e}"

    prompt = f"You are assisting with UI automation. Goal: '{goal_prompt}'. Return JSON with the key `element_text` indicating the on-screen text of the element to click."

    try:
        image_part = types.Part.from_bytes(data=screenshot_bytes, mime_type="image/png")
        response = call_gemini_with_retry(contents=[image_part, prompt])
        ai_response_text = (response.text or "").strip().replace("```json", "").replace("```", "")
        try:
            element_text = json.loads(ai_response_text).get("element_text")
        except Exception:
            element_text = None
        return element_text, None
    except Exception as e:
        return None, f"AI processing error: {e}"


def extract_links_by_clicking(driver, wait, status_callback=None):
    if status_callback:
        status_callback("info", " **Goal:** Link extraction.")

    extracted_links = {}
    links_to_find = ["Ustalenia ogólne", "Ustalenia morfoplastyczne", "Ustalenia szczegółowe", "Ustalenia końcowe"]
    original_window = driver.current_window_handle

    for label in links_to_find:
        link_locator = (By.XPATH, f"//td/div[text()='{label}']/parent::td/following-sibling::td//a")
        found_links = driver.find_elements(*link_locator)

        if found_links:
            link_to_click = found_links[0]
            try:
                driver.execute_script("arguments[0].click();", link_to_click)
                wait.until(EC.number_of_windows_to_be(2))
                new_window = [w for w in driver.window_handles if w != original_window][0]
                driver.switch_to.window(new_window)
                extracted_links[label] = driver.current_url

                if status_callback:
                    status_callback("success", f"Captured link: {label}")

                driver.close()
                driver.switch_to.window(original_window)
                time.sleep(1)
            except Exception as e:
                if status_callback:
                    status_callback("warning", f"Error clicking link for '{label}': {e}")
        else:
            pass

    return extracted_links


def analyze_documents_with_ai(_links_tuple, parcel_id, status_callback=None):
    links_dict = dict(_links_tuple)
    results = {'ogolne': {}, 'szczegolowe': {}}
    docs_content = {}

    for label, url in links_dict.items():
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()

            with fitz.open(stream=response.content, filetype="pdf") as doc:
                extracted_text = ""
                for page in doc:
                    page_text = page.get_text()
                    if page_text:
                        extracted_text += page_text + "\n"

                if len(extracted_text.strip()) < 100:
                    if status_callback:
                        status_callback("warning", f"Dokument '{label}' does not contain a text layer. Using OCR...")

                    extracted_text = ""

                    try:
                        import pytesseract
                        from PIL import Image
                        import io

                        for page_num, page in enumerate(doc, start=1):
                            pix = page.get_pixmap(matrix=fitz.Matrix(300 / 72, 300 / 72))
                            img_bytes = pix.tobytes("png")
                            img = Image.open(io.BytesIO(img_bytes))

                            page_text_ocr = pytesseract.image_to_string(img, lang='pol')
                            extracted_text += page_text_ocr + "\n"

                            if status_callback:
                                status_callback("info",
                                                f"OCR page {page_num}/{len(doc)} - extracted {len(page_text_ocr)} characters")

                        if extracted_text.strip():
                            if status_callback:
                                status_callback("success",
                                                f"OCR completed for '{label}' - {len(extracted_text)} characters")
                        else:
                            if status_callback:
                                status_callback("error", f"OCR detected no text in document '{label}'")

                    except ImportError:
                        if status_callback:
                            status_callback("error", "Missing pytesseract library.")
                        continue
                    except Exception as e:
                        if status_callback:
                            status_callback("error", f"OCR error for '{label}': {e}")
                        continue

                docs_content[label] = extracted_text.strip()

        except Exception as e:
            if status_callback:
                status_callback("error", f"Error processing '{label}': {e}")
            continue

    if "Ustalenia ogólne" in docs_content and docs_content["Ustalenia ogólne"]:
        prompt = f"You are a professional urban planner. Analyze the provided documents and provide the final report strictly in English.\n\n"
            f"Based on the following document, what is the overall objective and character of the area covered by this local plan?\n\n"
            f"Document:\n---\n{docs_content['Ustalenia ogólne']}"
        try:
            response = call_gemini_with_retry(contents=prompt)
            results['ogolne']['Cel Planu'] = (
                        client.models.generate_content(model=config.MODEL_NAME, contents=prompt).text or "").strip()
        except Exception as e:
            results['ogolne']['Cel Planu'] = f"Błąd AI: {e}"

    if "Ustalenia szczegółowe" in docs_content:
        doc_szczegolowe = docs_content["Ustalenia szczegółowe"]

        if doc_szczegolowe and len(doc_szczegolowe) > 50:
            prompt = f"""
You are a professional urban planner. Analyze the provided documents and provide the final report strictly in English.
	
	Analyze the text section titled "Ustalenia szczegółowe" (Detailed provisions).
Format the extracted data so it is human-readable (use bullet points and bold for key values) within the JSON string values.

Formatting guidance for each field:
1. **Oznaczenie Terenu (Area designation)**: Only the designation itself (e.g., **20.MN**).
2. **Przeznaczenie terenu (Land use)**: Use bullet points (hyphens). Separate primary from permissible uses. Bold key functions.
   Example:
   "- **Primary**: Residential development
- **Permissible**: Non-intrusive services"
3. **Wysokość zabudowy (Building height)**: If a range, use "from **X** to **Y** m". If storeys, e.g., "max **2** storeys".
4. **Wskaźniki zabudowy (Development parameters)**: One parameter per new line. Bold numeric values.
   Example:
   "- Max building coverage: **30%**
- Min biologically active area: **50%**"
5. **Geometria dachu (Roof geometry)**: List key features (pitch, orientation, covering).

Return the result STRICTLY as JSON with EXACTLY these keys (in Polish):
{{
  "Oznaczenie Terenu": "...",
  "Przeznaczenie terenu": "...",
  "Wysokość zabudowy": "...",
  "Wskaźniki zabudowy": "...",
  "Geometria dachu": "..."
}}

Document text:
---
{doc_szczegolowe[:25000]} 
"""
            try:
                response = call_gemini_with_retry(
                    contents=prompt,
                    config_params=types.GenerateContentConfig(response_mime_type="application/json")
                )
                parsed = None
                try:
                    parsed = json.loads((response.text or "").strip())
                except Exception:
                    parsed = None
                default_val = "No specific provisions in this excerpt"
                fields = [
                    "Oznaczenie Terenu",
                    "Przeznaczenie terenu",
                    "Wysokość zabudowy",
                    "Wskaźniki zabudowy",
                    "Geometria dachu",
                ]
                for key in fields:
                    if parsed and isinstance(parsed, dict) and key in parsed and parsed[key]:
                        results['szczegolowe'][key] = str(parsed[key]).strip()
                    else:
                        results['szczegolowe'][key] = default_val
            except Exception as e:
                for key in ["Oznaczenie Terenu", "Przeznaczenie terenu", "Wysokość zabudowy", "Wskaźniki zabudowy",
                            "Geometria dachu"]:
                    results['szczegolowe'][key] = f"Błąd AI: {e}"

        return results


@st.cache_data(show_spinner=False)
def fetch_raw_docs_cached(links_items: tuple):
    out = {}
    for label, url in links_items:
        try:
            resp = requests.get(url, timeout=60)
            resp.raise_for_status()
            ctype = resp.headers.get('Content-Type', 'application/octet-stream')
            path = urlparse(url).path.rsplit('/', 1)[-1] or None
            if not path:
                path = label.replace(' ', '_') + '.pdf'
            out[label] = {
                'url': url,
                'content': resp.content,
                'mime': ctype.split(';')[0],
                'filename': path
            }
        except Exception:
            continue
    return out


def run_ai_agent_flow(parcel_id, status_callback=None):
    options = webdriver.ChromeOptions()
    options.add_argument("--headless=new")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-gpu")
    options.add_argument("--window-size=1920,1080")
    options.add_argument("--force-device-scale-factor=1")

    if os.getenv('CHROME_BIN'):
        options.binary_location = os.getenv('CHROME_BIN')
        service = Service(os.getenv('CHROMEDRIVER_PATH'))
    else:
        service = Service()

    driver = webdriver.Chrome(service=service, options=options)
    final_results = {}

    try:
        driver.get("https://mapa.szczecin.eu/gpt4/?permalink=56520129")
        time.sleep(5)
        wait = WebDriverWait(driver, 20)

        search_box = wait.until(EC.element_to_be_clickable((By.XPATH, "//input[@placeholder='Szukaj...']")))
        search_box.send_keys(parcel_id)
        wait.until(EC.visibility_of_element_located((By.XPATH, "//li[contains(@class, 'x-boundlist-item')]")))
        time.sleep(1);
        search_box.send_keys(Keys.RETURN)
        time.sleep(1);
        search_box.send_keys(Keys.RETURN)

        if status_callback:
            status_callback("success", "Step 1/3: Parcel located.")

        time.sleep(4)
        ActionChains(driver).move_by_offset(driver.get_window_size()['width'] / 2,
                                            driver.get_window_size()['height'] / 2).context_click().perform()

        if status_callback:
            status_callback("success", "Step 2/3: Context menu opened.")

        time.sleep(1)
        try:
            wait.until(EC.element_to_be_clickable((By.XPATH, "//*[contains(text(), 'Informacje o obiekcie')]"))).click()
            if status_callback:
                status_callback("success", "Action 'Object information' executed.")
            time.sleep(3)
        except Exception as e:
            if status_callback:
                status_callback("error", f"Failed to open the 'Object information' window: {e}")
            raise e

        if status_callback:
            status_callback("info", "Step 3/3: Checking MPZP status in the dedicated window...")

        time.sleep(2)

        info_window_context_xpath = "//div[contains(@class, 'x-window') and .//span[text()='Informacje o obiekcie']]"
        mpzp_uchwalony_locator = (By.XPATH,
                                  info_window_context_xpath + "//*[contains(text(), 'MPZP - Tereny elementarne')]")
        mpzp_wszczety_locator = (By.XPATH, info_window_context_xpath + "//*[contains(text(), 'MPZP - plany wszczęte')]")

        if driver.find_elements(*mpzp_uchwalony_locator):
            if status_callback:
                status_callback("success", "Found an ADOPTED MPZP for this parcel. Continuing analysis...")

            try:
                driver.find_element(*mpzp_uchwalony_locator).click()
                time.sleep(2)

                final_links = extract_links_by_clicking(driver, wait, status_callback)

                if final_links:
                    final_results['links'] = final_links

                    if status_callback:
                        status_callback("info", "Starting the AI Analyst Agent...")

                    analysis = analyze_documents_with_ai(tuple(sorted(final_links.items())), parcel_id, status_callback)
                    if analysis: final_results['analysis'] = analysis
                else:
                    if status_callback:
                        status_callback("error",
                                        "Failed to extract any links even though MPZP was identified.")
            except Exception as e:
                if status_callback:
                    status_callback("error", f"An error occurred during interaction with the existing MPZP: {e}")
                return {}

        elif driver.find_elements(*mpzp_wszczety_locator):
            if status_callback:
                status_callback("warning",
                                "For this parcel, the MPZP preparation procedure has been initiated, but the plan has not yet been adopted.")
                status_callback("info", "The agent is terminating, as there are no final documents to analyze yet.")
            return {"status": "wszczęty"}

        else:
            if status_callback:
                status_callback("error",
                                "No MPZP data was found in the information window for the selected parcel.")
                status_callback("info", "The agent is terminating.")
            return {"status": "brak"}

    finally:
        driver.quit()

    return final_results

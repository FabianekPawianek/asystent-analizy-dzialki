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
import vertexai
from vertexai.generative_models import GenerativeModel, Part
from langchain_google_vertexai import VertexAIEmbeddings, VertexAI
import config

generative_model = None
llm = None

def init_ai(project_id):
    global generative_model, llm
    vertexai.init(project=project_id, location=config.LOCATION)
    generative_model = GenerativeModel(config.MODEL_NAME)
    llm = VertexAI(model_name=config.MODEL_NAME)

def perform_ai_step(driver, goal_prompt, status_callback=None):
    if status_callback:
        status_callback("info", f" **Cel:** {goal_prompt}")
        
    screenshot_bytes = driver.get_screenshot_as_png()
    prompt = f"Cel: '{goal_prompt}'. Odpowiedz w JSON, podając `element_text` do kliknięcia."
    
    try:
        response = generative_model.generate_content([Part.from_data(screenshot_bytes, mime_type="image/png"), prompt])
        ai_response_text = response.text.strip().replace("```json", "").replace("```", "")
        return json.loads(ai_response_text).get("element_text"), None
    except Exception as e:
        return None, f"Błąd przetwarzania AI: {e}"

def extract_links_by_clicking(driver, wait, status_callback=None):
    if status_callback:
        status_callback("info", " **Cel:** Błyskawiczna ekstrakcja linków.")
        
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
                    status_callback("success", f"Pobrano link: {label}")
                    
                driver.close()
                driver.switch_to.window(original_window)
                time.sleep(1)
            except Exception as e:
                if status_callback:
                    status_callback("warning", f"Błąd podczas klikania w link dla '{label}': {e}")
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
                        status_callback("warning", f"Dokument '{label}' nie zawiera warstwy tekstowej. Używam OCR...")
                    
                    extracted_text = ""

                    try:
                        import pytesseract
                        from PIL import Image
                        import io

                        for page_num, page in enumerate(doc, start=1):
                            pix = page.get_pixmap(matrix=fitz.Matrix(300/72, 300/72))
                            img_bytes = pix.tobytes("png")
                            img = Image.open(io.BytesIO(img_bytes))

                            page_text_ocr = pytesseract.image_to_string(img, lang='pol')
                            extracted_text += page_text_ocr + "\n"

                            if status_callback:
                                status_callback("info", f"OCR strona {page_num}/{len(doc)} - wyodrębniono {len(page_text_ocr)} znaków")

                        if extracted_text.strip():
                            if status_callback:
                                status_callback("success", f"OCR zakończone dla '{label}' - {len(extracted_text)} znaków")
                        else:
                            if status_callback:
                                status_callback("error", f"OCR nie wykrył tekstu w dokumencie '{label}'")

                    except ImportError:
                        if status_callback:
                            status_callback("error", "Brak biblioteki pytesseract.")
                        continue
                    except Exception as e:
                        if status_callback:
                            status_callback("error", f"Błąd OCR dla '{label}': {e}")
                        continue

                docs_content[label] = extracted_text.strip()

        except Exception as e:
            if status_callback:
                status_callback("error", f"Błąd podczas przetwarzania '{label}': {e}")
            continue

    if "Ustalenia ogólne" in docs_content and docs_content["Ustalenia ogólne"]:
        prompt = f"Na podstawie tego dokumentu, jaki jest ogólny cel i charakter obszaru objętego tym planem?\n\nDokument:\n---\n{docs_content['Ustalenia ogólne']}"
        results['ogolne']['Cel Planu'] = llm.invoke(prompt)

    if "Ustalenia szczegółowe" in docs_content:
        doc_szczegolowe = docs_content["Ustalenia szczegółowe"]

        if doc_szczegolowe and len(doc_szczegolowe) > 50:
            id_prompt = f"Na podstawie poniższego tekstu z dokumentu 'Ustalenia szczegółowe', jaki jest symbol/oznaczenie terenu elementarnego? (np. 'S.N.9006.MC'). Odpowiedz tylko samym symbolem terenu.\n\nTekst dokumentu:\n---\n{doc_szczegolowe[:5000]}"
            results['szczegolowe']['Oznaczenie Terenu'] = llm.invoke(id_prompt)

            detail_questions = {
                "Przeznaczenie terenu": "Jakie jest szczegółowe przeznaczenie terenu (podstawowe i dopuszczalne) oraz jakie są zakazy?",
                "Wysokość zabudowy": "Jakie są szczegółowe ustalenia dotyczące wysokości zabudowy w metrach?",
                "Wskaźniki zabudowy": "Jakie są szczegółowe wskaźniki, takie jak maksymalna powierzchnia zabudowy i minimalna powierzchnia biologicznie czynna?",
                "Geometria dachu": "Jakie są szczegółowe wymagania dotyczące geometrii dachu i jego pokrycia?",
            }

            for key, question in detail_questions.items():
                prompt = f"Na podstawie TYLKO i WYŁĄCZNIE poniższego dokumentu 'Ustalenia szczegółowe', odpowiedz na pytanie: {question}\n\nDokument:\n---\n{doc_szczegolowe}"
                results['szczegolowe'][key] = llm.invoke(prompt)

    return results

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
        time.sleep(1); search_box.send_keys(Keys.RETURN)
        time.sleep(1); search_box.send_keys(Keys.RETURN)
        
        if status_callback:
            status_callback("success", "Krok 1/3: Działka zlokalizowana.")
            
        time.sleep(4)
        ActionChains(driver).move_by_offset(driver.get_window_size()['width'] / 2, driver.get_window_size()['height'] / 2).context_click().perform()
        
        if status_callback:
            status_callback("success", "Krok 2/3: Menu kontekstowe otwarte.")
            
        time.sleep(1)
        try:
            wait.until(EC.element_to_be_clickable((By.XPATH, "//*[contains(text(), 'Informacje o obiekcie')]"))).click()
            if status_callback:
                status_callback("success", "Akcja 'Informacje o obiekcie' wykonana.")
            time.sleep(3)
        except Exception as e:
            if status_callback:
                status_callback("error", f"Nie udało się otworzyć okna 'Informacje o obiekcie': {e}")
            raise e

        if status_callback:
            status_callback("info", "Krok 3/3: Sprawdzanie statusu MPZP w dedykowanym oknie...")
            
        time.sleep(2)

        info_window_context_xpath = "//div[contains(@class, 'x-window') and .//span[text()='Informacje o obiekcie']]"
        mpzp_uchwalony_locator = (By.XPATH, info_window_context_xpath + "//*[contains(text(), 'MPZP - Tereny elementarne')]")
        mpzp_wszczety_locator = (By.XPATH, info_window_context_xpath + "//*[contains(text(), 'MPZP - plany wszczęte')]")

        if driver.find_elements(*mpzp_uchwalony_locator):
            if status_callback:
                status_callback("success", "Znaleziono UCHWALONY MPZP dla tej działki. Kontynuuję analizę...")
                
            try:
                driver.find_element(*mpzp_uchwalony_locator).click()
                time.sleep(2)

                final_links = extract_links_by_clicking(driver, wait, status_callback)

                if final_links:
                    final_results['links'] = final_links

                    if status_callback:
                        status_callback("info", "Uruchamiam Agenta Analityka AI...")
                        
                    analysis = analyze_documents_with_ai(tuple(sorted(final_links.items())), parcel_id, status_callback)
                    if analysis: final_results['analysis'] = analysis
                else:
                    if status_callback:
                        status_callback("error", "Nie udało się wyodrębnić żadnych linków, mimo że MPZP został zidentyfikowany.")
            except Exception as e:
                if status_callback:
                    status_callback("error", f"Wystąpił błąd na etapie interakcji z istniejącym MPZP: {e}")
                return {}

        elif driver.find_elements(*mpzp_wszczety_locator):
            if status_callback:
                status_callback("warning", "Dla tej działki procedura sporządzenia MPZP została wszczęta, ale plan nie jest jeszcze uchwalony.")
                status_callback("info", "Agent kończy pracę, ponieważ nie ma jeszcze finalnych dokumentów do analizy.")
            return {"status": "wszczęty"}

        else:
            if status_callback:
                status_callback("error", "Dla wybranej działki w oknie informacyjnym nie znaleziono żadnych danych o MPZP.")
                status_callback("info", "Agent kończy pracę.")
            return {"status": "brak"}

    finally:
        driver.quit()

    return final_results

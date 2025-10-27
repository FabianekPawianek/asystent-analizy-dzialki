import streamlit as st
import time
import json
import requests
import fitz
import folium
from streamlit_folium import st_folium
from pyproj import Transformer
from urllib.parse import quote_plus
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

PROJECT_ID = os.getenv("GCP_PROJECT_ID")
LOCATION = "us-central1"
MODEL_NAME = "gemini-2.5-pro"
EMBEDDING_MODEL_NAME = "text-embedding-004"
vertexai.init(project=PROJECT_ID, location=LOCATION)
generative_model = GenerativeModel(MODEL_NAME)
embeddings_model = VertexAIEmbeddings(model_name=EMBEDDING_MODEL_NAME)
llm = VertexAI(model_name=MODEL_NAME)


def geocode_address_to_coords(address):
    nominatim_url = f"https://nominatim.openstreetmap.org/search?q={quote_plus(address)}&format=json&limit=1&countrycodes=pl"
    headers = {'User-Agent': 'AsystentAnalizyDzialki'}
    response = requests.get(nominatim_url, headers=headers, timeout=15);
    response.raise_for_status()
    geodata = response.json()
    if not geodata: return None, "Nie znaleziono współrzędnych dla podanego adresu."
    return (float(geodata[0]['lat']), float(geodata[0]['lon'])), None


def get_parcel_by_id(parcel_id):
    uldk_url = f"https://uldk.gugik.gov.pl/?request=GetParcelById&id={parcel_id}&result=geom_wkt"
    response = requests.get(uldk_url, timeout=15);
    response.raise_for_status()
    responseText = response.text.strip()
    if responseText.startswith(
        '-1'): return None, f"Błąd wewnętrzny: Nie znaleziono danych dla działki o ID: {parcel_id}."
    wkt_geom_raw = responseText.split('\n')[1].strip()
    if 'SRID=' in wkt_geom_raw: wkt_geom_raw = wkt_geom_raw.split(';', 1)[1]
    coords_str = wkt_geom_raw.replace('POLYGON((', '').replace('))', '')
    coords_pairs = [pair.split() for pair in coords_str.split(',')]
    coords_2180 = [[float(x), float(y)] for x, y in coords_pairs]
    return {"ID Działki": parcel_id, "Współrzędne EPSG:2180": coords_2180}, None


def get_parcel_from_coords(lat, lon):
    try:
        transformer = Transformer.from_crs("EPSG:4326", "EPSG:2180", always_xy=True)
        x, y = transformer.transform(lon, lat)
        identify_url = f"https://uldk.gugik.gov.pl/?request=GetParcelByXY&xy={x},{y}&result=id"
        response_id = requests.get(identify_url, timeout=15);
        response_id.raise_for_status()
        id_text = response_id.text.strip()
        if id_text.startswith('-1') or len(id_text.split(
            '\n')) < 2: return None, "W tym miejscu nie zidentyfikowano działki. Spróbuj kliknąć precyzyjniej."
        parcel_id = id_text.split('\n')[1].strip()
        return get_parcel_by_id(parcel_id)
    except Exception as e:
        return None, f"Błąd identyfikacji działki: {e}"


def transform_coordinates_to_wgs84(coords_2180):
    transformer = Transformer.from_crs("EPSG:2180", "EPSG:4326", always_xy=True)
    return [[transformer.transform(x, y)[1], transformer.transform(x, y)[0]] for x, y in coords_2180]


def perform_ai_step(driver, model, goal_prompt):
    st.info(f"🎯 **Cel:** {goal_prompt}")
    screenshot_bytes = driver.get_screenshot_as_png()
    prompt = f"Cel: '{goal_prompt}'. Odpowiedz w JSON, podając `element_text` do kliknięcia."
    response = model.generate_content([Part.from_data(screenshot_bytes, mime_type="image/png"), prompt])
    try:
        ai_response_text = response.text.strip().replace("```json", "").replace("```", "")
        return json.loads(ai_response_text).get("element_text"), None
    except Exception as e:
        return None, f"Błąd przetwarzania AI: {e}. Odpowiedź: {response.text}"


def extract_links_by_clicking(driver, wait):
    st.info("🎯 **Cel:** Błyskawiczna ekstrakcja linków.")
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
                st.success(f"Pobrano link: {label}")
                driver.close()
                driver.switch_to.window(original_window)
                time.sleep(1)
            except Exception as e:
                st.warning(f"Błąd podczas klikania w link dla '{label}': {e}")
        else:
            st.write(f"ℹ️ Link dla '{label}' nie istnieje na stronie. Pomijam.")

    return extracted_links


@st.cache_data
def analyze_documents_with_ai(_links_tuple, parcel_id):
    links_dict = dict(_links_tuple)
    results = {'ogolne': {}, 'szczegolowe': {}}
    docs_content = {}
    for label, url in links_dict.items():
        try:
            response = requests.get(url);
            response.raise_for_status()
            with fitz.open(stream=response.content, filetype="pdf") as doc:
                docs_content[label] = "".join(page.get_text() for page in doc)
        except Exception:
            continue
    if "Ustalenia ogólne" in docs_content:
        prompt = f"Na podstawie tego dokumentu, jaki jest ogólny cel i charakter obszaru objętego tym planem?\n\nDokument:\n---\n{docs_content['Ustalenia ogólne']}"
        results['ogolne']['Cel Planu'] = llm.invoke(prompt)
    if "Ustalenia szczegółowe" in docs_content:
        doc_szczegolowe = docs_content["Ustalenia szczegółowe"]
        id_prompt = f"Na podstawie poniższego tekstu, jaki jest symbol/oznaczenie terenu elementarnego? (np. 'S.N.9006.MC')\n\nTekst:\n---\n{doc_szczegolowe[:1000]}"
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


def run_ai_agent_flow(parcel_id):
    service = Service()
    options = webdriver.ChromeOptions()
    options.add_argument("--headless=new")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-gpu")
    options.add_argument("--window-size=1920,1080")
    options.add_argument("--force-device-scale-factor=1")
    driver = webdriver.Chrome(service=service, options=options)
    final_results = {}
    try:
        with st.expander("Postęp misji agenta nawigacyjnego", expanded=True):
            driver.get("https://mapa.szczecin.eu/gpt4/?permalink=56520129")
            time.sleep(5)
            wait = WebDriverWait(driver, 20)

            search_box = wait.until(EC.element_to_be_clickable((By.XPATH, "//input[@placeholder='Szukaj...']")))
            search_box.send_keys(parcel_id)
            wait.until(EC.visibility_of_element_located((By.XPATH, "//li[contains(@class, 'x-boundlist-item')]")))
            time.sleep(1); search_box.send_keys(Keys.RETURN)
            time.sleep(1); search_box.send_keys(Keys.RETURN)
            st.success("✅ Krok 1/3: Działka zlokalizowana.")
            time.sleep(4)
            ActionChains(driver).move_by_offset(driver.get_window_size()['width'] / 2, driver.get_window_size()['height'] / 2).context_click().perform()
            st.success("✅ Krok 2/3: Menu kontekstowe otwarte.")
            time.sleep(1)
            try:
                wait.until(EC.element_to_be_clickable((By.XPATH, "//*[contains(text(), 'Informacje o obiekcie')]"))).click()
                st.success("✅ Akcja 'Informacje o obiekcie' wykonana.")
                time.sleep(3)
            except Exception as e:
                st.error(f"⚠️ Nie udało się otworzyć okna 'Informacje o obiekcie': {e}")
                raise e

            st.info("🔎 Krok 3/3: Sprawdzanie statusu MPZP w dedykowanym oknie...")
            time.sleep(2)

            info_window_context_xpath = "//div[contains(@class, 'x-window') and .//span[text()='Informacje o obiekcie']]"
            mpzp_uchwalony_locator = (By.XPATH, info_window_context_xpath + "//*[contains(text(), 'MPZP - Tereny elementarne')]")
            mpzp_wszczety_locator = (By.XPATH, info_window_context_xpath + "//*[contains(text(), 'MPZP - plany wszczęte')]")

            if driver.find_elements(*mpzp_uchwalony_locator):
                st.success("✅ Znaleziono UCHWALONY MPZP dla tej działki. Kontynuuję analizę...")
                try:
                    driver.find_element(*mpzp_uchwalony_locator).click()
                    time.sleep(2)

                    with st.spinner("Nawigacja zakończona. Ekstrakcja linków..."):
                        final_links = extract_links_by_clicking(driver, wait)

                    if final_links:
                        final_results['links'] = final_links
                        st.subheader("Pobrane Dokumenty:"); st.toast("✅ Linki pobrane!")
                        for label, link in final_links.items(): st.markdown(f"**{label}:** [Otwórz]({link})")
                        with st.spinner("Uruchamiam Agenta Analityka AI..."):
                            analysis = analyze_documents_with_ai(tuple(sorted(final_links.items())), parcel_id)
                        if analysis: final_results['analysis'] = analysis
                    else:
                        st.error("Nie udało się wyodrębnić żadnych linków, mimo że MPZP został zidentyfikowany.")
                except Exception as e:
                    st.error(f"Wystąpił błąd na etapie interakcji z istniejącym MPZP: {e}")
                    return {}

            elif driver.find_elements(*mpzp_wszczety_locator):
                st.warning("🔵 Dla tej działki procedura sporządzenia MPZP została wszczęta, ale plan nie jest jeszcze uchwalony.")
                st.info("Agent kończy pracę, ponieważ nie ma jeszcze finalnych dokumentów do analizy.")
                return {"status": "wszczęty"}

            else:
                st.error("❌ Dla wybranej działki w oknie informacyjnym nie znaleziono żadnych danych o MPZP.")
                st.info("Agent kończy pracę.")
                return {"status": "brak"}

    finally:
        st.write("Zamykam przeglądarkę.")
        driver.quit()

    return final_results


st.set_page_config(layout="wide");
st.title("Asystent Analizy Działki w Szczecinie - Wersja Beta 0.1")
st.caption("Autor: Fabian Korycki | Powered by Google Gemini AI")
st.markdown("---")
for key in ['map_center', 'parcel_data', 'analysis_results']:
    if key not in st.session_state: st.session_state[key] = None
st.header("Krok 1: Wyszukaj i wybierz działkę na mapie")
with st.form(key="address_form"):
    address_input = st.text_input("Wpisz adres lub współrzędne i wciśnij Enter:")
    submitted = st.form_submit_button("Pokaż mapę do wyboru")
if submitted:
    st.session_state.parcel_data = None;
    st.session_state.analysis_results = None
    with st.spinner("Pobieram współrzędne..."):
        coords, error = geocode_address_to_coords(address_input)
        if error:
            st.error(error); st.session_state.map_center = None
        else:
            st.session_state.map_center = coords
if st.session_state.map_center and not st.session_state.parcel_data:
    st.info("KLIKNIJ na mapie, aby precyzyjnie wybrać interesującą Cię działkę.")
    m = folium.Map(location=st.session_state.map_center, zoom_start=18)
    folium.TileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
                     attr='Esri', name='Satelita', overlay=True).add_to(m)
    folium.WmsTileLayer(url="https://integracja.gugik.gov.pl/cgi-bin/KrajowaIntegracjaEwidencjiGruntow",
                        layers="dzialki,numery_dzialek", transparent=True, fmt="image/png",
                        name="Działki Ewidencyjne").add_to(m)
    folium.LayerControl().add_to(m)
    map_data = st_folium(m, use_container_width=True, height=500)

    if map_data and map_data.get("last_clicked"):
        lat, lon = map_data["last_clicked"]["lat"], map_data["last_clicked"]["lng"]
        with st.spinner(f"Identyfikuję działkę..."):
            parcel_data, error = get_parcel_from_coords(lat, lon)
            if error:
                st.error(error)
            else:
                st.session_state.parcel_data = parcel_data; st.rerun()
if st.session_state.parcel_data:
    st.success("✅ Pomyślnie wybrano działkę!")
    col1, col2 = st.columns([1, 2])
    with col1:
        st.subheader("Dane ewidencyjne");
        st.markdown("**Numer działki:**");
        st.code(st.session_state.parcel_data['ID Działki'])
    with col2:
        st.subheader("Potwierdzenie wizualne")
        coords_wgs84 = transform_coordinates_to_wgs84(st.session_state.parcel_data["Współrzędne EPSG:2180"])
        map_center = [sum(p[0] for p in coords_wgs84) / len(coords_wgs84),
                      sum(p[1] for p in coords_wgs84) / len(coords_wgs84)]
        m_confirm = folium.Map(location=map_center, zoom_start=18)
        folium.TileLayer(
            'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
            attr='Esri').add_to(m_confirm)
        folium.Polygon(locations=coords_wgs84, color='red', fill_opacity=0.4,
                       tooltip=st.session_state.parcel_data['ID Działki']).add_to(m_confirm)
        st_folium(m_confirm, use_container_width=True, height=400)

    st.markdown("---")
    st.header("Krok 2: Uruchom analizę AI dla wybranej działki")
    if st.button("Rozpocznij pełną analizę AI"):
        results = run_ai_agent_flow(st.session_state.parcel_data['ID Działki'])
        st.session_state.analysis_results = results

if st.session_state.analysis_results:
    st.success("🎉 Misja Agenta Analityka zakończona!")
    results = st.session_state.analysis_results
    if 'analysis' in results and results['analysis']:
        if 'ogolne' in results['analysis'] and results['analysis']['ogolne']:
            st.subheader("Analiza Ogólna Planu (MPZP)")
            st.markdown(f"**Cel Planu:**");
            st.info(f"{results['analysis']['ogolne'].get('Cel Planu', 'Brak danych.')}")
        if 'szczegolowe' in results['analysis'] and results['analysis']['szczegolowe']:
            st.subheader(
                f"Analiza Szczegółowa dla Terenu: {results['analysis']['szczegolowe'].get('Oznaczenie Terenu', 'N/A')}")
            for key, value in results['analysis']['szczegolowe'].items():
                if key != 'Oznaczenie Terenu': st.markdown(f"**{key}:**"); st.info(f"{value}")
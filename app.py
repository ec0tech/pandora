# app.py içeriği (Flask Backend)

from flask import Flask, render_template, request
import os
import requests
from dotenv import load_dotenv
from google import genai
from google.genai.errors import APIError

# --- GİZLİ ANAHTARLARIN YÜKLENMESİ ---
load_dotenv()
TMDB_API_KEY = os.getenv("TMDB_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# --- FLASK UYGULAMASI ---
app = Flask(__name__)

# --- API VE AI YAPILANDIRMASI ---
client = None
if GEMINI_API_KEY:
    try:
        # API anahtarı doğruysa, Gemini istemcisini başlat
        client = genai.Client(api_key=GEMINI_API_KEY)
        MODEL_NAME = 'gemini-2.5-flash'
    except Exception as e:
        print(f"HATA: Gemini istemcisi başlatılamadı. {e}")

TMDB_IMAGE_BASE_URL = "https://image.tmdb.org/t/p/w500" 

# TMDb için Tür ID Haritası (Türkçe isimlerle)
GENRE_ID_MAP = {
    "Aksiyon": 28, "Macera": 12, "Animasyon": 16, "Komedi": 35,
    "Suç": 80, "Belgesel": 99, "Dram": 18, "Aile": 10751,
    "Fantastik": 14, "Tarih": 36, "Korku": 27, "Müzik": 10402,
    "Gizem": 9648, "Romantik": 10749, "Bilim Kurgu": 878, "Gerilim": 53, 
    "Savaş": 10752, "Western": 37, "Tüm Türler (Popüler)": 0 # Tüm popüler filmler için özel ID
}

# --- TMDb'den veri çekme fonksiyonu ---
def get_movies_from_tmdb(genre_name, film_filtresi):
    genre_id = GENRE_ID_MAP.get(genre_name)
    
    TMDB_URL = "https://api.themoviedb.org/3/discover/movie"
    
    # Tüm popüler filmleri çekmek için özel durum
    if genre_id == 0:
        TMDB_URL = "https://api.themoviedb.org/3/movie/popular"
        params = {'api_key': TMDB_API_KEY, 'language': 'en-US'}
    else:
        # Normal filtreleme mantığı
        params = {
            'api_key': TMDB_API_KEY,               
            'with_genres': genre_id,               
            'sort_by': 'popularity.desc', 
            'vote_average.gte': 6.5,      
            'language': 'en-US',
        }
        # "Az bilinen" gibi özel filtre varsa, filtreleri daralt
        if "az bilinen" in film_filtresi.lower():
             params['vote_count.lte'] = 1000  
             params['vote_average.gte'] = 7.0 
    
    try:
        response = requests.get(TMDB_URL, params=params)
        response.raise_for_status() 

        movies_list = response.json().get('results', [])[:20] # 20 film çekiyoruz
        
        context_data = []
        for movie in movies_list:
            context_data.append({
                "title": movie.get('title'),
                "year": movie.get('release_date', 'Bilinmiyor')[:4], 
                "overview": movie.get('overview'),
                "rating": movie.get('vote_average'),
                "poster_url": TMDB_IMAGE_BASE_URL + movie.get('poster_path', '') 
            })
        return context_data

    except requests.RequestException as e:
        return f"TMDb API isteği hatası: {e}"

# --- PROMPT OLUŞTURMA VE GEMINI'YE GÖNDERME FONKSİYONLARI ---
def get_gemini_recommendation(genre, film_filtresi, movie_context):
    
    movies_string = "\n".join([
        f"Title: {m['title']}, Year: {m['year']}, Overview: {m['overview']}, Rating: {m['rating']}"
        for m in movie_context
    ])

    prompt = f"""
You are an expert AI movie critic named 'Picky Cinephile'. Provide exactly 3 personalized movie recommendations.

**GIVEN DATA SOURCE:**
The movies listed below are sourced from TMDb. You must ONLY recommend films from this list.

**USER PREFERENCES:**
Genre: {genre}
Specific Filter/Mood: {film_filtresi}

**PROVIDED MOVIE LIST (Analyze the Overview, Title, and Rating to match the filter):**
---
{movies_string}
---

**RECOMMENDATION RULES:**
1.  Suggest **exactly 3 movies** from the provided list.
2.  Each suggestion must perfectly satisfy the **Specific Filter/Mood** ({film_filtresi}).
3.  Focus the selection on movies that contain {film_filtresi} elements (e.g., strong plot twists, positive mood, emotional depth, etc.).
4.  Output MUST STRICTLY follow the OUTPUT FORMAT. Do NOT include the Poster URL in the text output.

**OUTPUT FORMAT:**
Movies:
1.  **[Movie Title]** ([Year]) - [Genre] | Puan: [Rating]: [Brief Explanation on Fit]
2.  **[Movie Title]** ([Year]) - [Genre] | Puan: [Rating]: [Brief Explanation on Fit]
3.  **[Movie Title]** ([Year]) - [Genre] | Puan: [Rating]: [Brief Explanation on Fit]
"""
    if not client:
        return "Yapay zeka servisi kapalıdır."
    
    try:
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt
        )
        return response.text
    except Exception as e:
        # Eğer çok uzun bir prompt olursa buraya düşebilir.
        return f"Gemini API Hatası: {e}. Lütfen filtreyi kısaltın veya TMDb API anahtarınızı kontrol edin."


# --- YANITI WEB İÇİN İŞLEME FONKSİYONU ---

def format_recommendation_for_web(raw_text, full_movie_list):
    """Gemini'den gelen düz metin yanıtını, afiş URL'lerini içeren HTML kartlarına dönüştürür."""
    
    lines = raw_text.split('\n')
    
    # Eğer Gemini hata döndürdüyse
    if not lines or "Gemini API Hatası" in raw_text:
        return f"<p class='error'>Öneri alınamadı: {raw_text}</p>"
        
    html_output = "<div class='recommendation-grid'>"
    
    for line in lines:
        if line.startswith(('1.', '2.', '3.')):
            
            # Filmin adını çekme
            movie_title = None
            try:
                title_start = line.find('**') + 2
                title_end = line.find('**', title_start)
                movie_title = line[title_start:title_end].strip()
            except:
                continue

            # Tam filmi (afiş URL'si ile) listede ara
            movie_data = next((m for m in full_movie_list if m['title'] == movie_title), None)
            
            # HTML Kartını Oluştur
            if movie_data:
                # Açıklama kısmı
                explanation = line.split(']:')[-1].strip()
                
                # Resim yolu yoksa varsayılan resim koy
                poster_src = movie_data['poster_url'] if movie_data['poster_url'].endswith('null') == False else 'static/placeholder.png'
                
                html_output += f"""
                <div class='movie-card'>
                    <img src='{poster_src}' alt='{movie_title} Poster'>
                    <div class='card-content'>
                        <h3>{movie_title} ({movie_data['year']})</h3>
                        <p class='rating'>⭐ Puan: {movie_data['rating']}</p>
                        <p class='explanation'>{explanation}</p>
                    </div>
                </div>
                """
    
    html_output += "</div>"
    return html_output

# --- FLASK WEB ROTALARI ---

@app.route('/', methods=['GET', 'POST'])
def index():
    recommendations_html = None
    
    if request.method == 'POST':
        user_genre = request.form['genre']
        user_filter = request.form['film_filtresi']
        
        # 1. TMDb'den filmleri çek
        movie_context = get_movies_from_tmdb(user_genre, user_filter)
        
        if isinstance(movie_context, str):
            recommendations_html = f"<p class='error'>{movie_context}</p>"
        else:
            # 2. Gemini'den öneri al
            gemini_response = get_gemini_recommendation(user_genre, user_filter, movie_context)
            
            # 3. Yanıtı işle ve HTML'e dönüştür
            recommendations_html = format_recommendation_for_web(gemini_response, movie_context)

    # index.html dosyasını göster
    return render_template('index.html', 
                           recommendations=recommendations_html,
                           genres=GENRE_ID_MAP.keys())

# --- SUNUCUYU ÇALIŞTIRMA (Render bu kısmı kullanmaz, Procfile kullanır) ---
if __name__ == '__main__':
    app.run(debug=True)
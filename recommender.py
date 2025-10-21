import os
import requests
from dotenv import load_dotenv
from google import genai
from google.genai.errors import APIError

# --- GİZLİ ANAHTARLARIN YÜKLENMESİ ---
load_dotenv()
TMDB_API_KEY = os.getenv("TMDB_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# --- GEMINI AYARLARI ---
client = None
if GEMINI_API_KEY:
    try:
        client = genai.Client(api_key=GEMINI_API_KEY)
        MODEL_NAME = 'gemini-2.5-flash'
        print("Gemini istemcisi başarıyla başlatıldı.")
    except Exception as e:
        print(f"HATA: Gemini istemcisi başlatılamadı. {e}")
else:
    print("HATA: GEMINI_API_KEY bulunamadı. Lütfen .env dosyasını kontrol edin.")

# TMDb, afiş resimlerinin başına eklenmesi gereken URL
TMDB_IMAGE_BASE_URL = "https://image.tmdb.org/t/p/w500" 

# --- TEST AMAÇLI KARMAŞIK KULLANICI GİRDİLERİ ---
# Kullanıcıdan alacağınız iki temel girdi
KULLANICI_GENRE_ADI = "Drama"   # Örn: Dram türünde
KULLANICI_FILTRESI = "Plot twist içeren, az bilinen ama yüksek puan almış" # Örn: Karmaşık Filtre
# Not: Modeli test ederken farklı türler (Comedy, Animation) ve filtreler (az bilinen, neşeli, gergin) deneyebilirsiniz.


# TMDb'nin kullandığı Tür ID'leri
GENRE_ID_MAP = {
    "Action": 28, "Adventure": 12, "Animation": 16, "Comedy": 35,
    "Crime": 80, "Documentary": 99, "Drama": 18, "Family": 10751,
    "Fantasy": 14, "History": 36, "Horror": 27, "Music": 10402,
    "Mystery": 9648, "Romance": 10749, "Sci-Fi": 878, "Thriller": 53, 
    "War": 10752, "Western": 37
}

# --- BİRLEŞİK VERİ ÇEKME FONKSİYONU (RAG'in Başlangıcı) ---
def get_movies_from_tmdb(genre_name, film_filtresi):
    """
    TMDb Discover API'sini kullanarak filme özel filtrelerle (az bilinen, yüksek puan) 
    15 filmi çeker ve afiş URL'lerini de ekler.
    """
    genre_id = GENRE_ID_MAP.get(genre_name)
    if not genre_id:
        return f"Hata: {genre_name} türü haritada bulunamadı."

    TMDB_URL = "https://api.themoviedb.org/3/discover/movie"
    
    # Filtre mantığı:
    # Plot twist için 'vote_count' az ve 'vote_average' yüksek yapmayız. 
    # 'Az bilinen ama beğenilmiş' için 'vote_count' düşük ve 'vote_average' yüksek olmalı.
    
    params = {
        'api_key': TMDB_API_KEY,               
        'with_genres': genre_id,               
        'sort_by': 'vote_average.desc',  # Puanı yüksek olanları öne çıkar
        'vote_count.gte': 500 if "az bilinen" not in film_filtresi.lower() else 50, # Az bilinen ise oyu az olanları da al
        'vote_average.gte': 7.5 if "az bilinen" in film_filtresi.lower() else 6.5, # Puanı yüksek olanları öne çıkar
        'language': 'en-US'                    
    }

    try:
        response = requests.get(TMDB_URL, params=params)
        response.raise_for_status() 

        movies_list = response.json().get('results', [])[:15] # İlk 15 filmi alıyoruz
        
        context_data = []
        for movie in movies_list:
            context_data.append({
                "title": movie.get('title'),
                "year": movie.get('release_date', 'Bilinmiyor')[:4], 
                "overview": movie.get('overview'),
                "rating": movie.get('vote_average'),
                # Web sitesinde afişi göstermek için gereken URL
                "poster_url": TMDB_IMAGE_BASE_URL + movie.get('poster_path', '') 
            })
        return context_data

    except requests.RequestException as e:
        return f"TMDb API isteği hatası: {e}"

# --- PROMPT OLUŞTURMA VE GEMINI'YE GÖNDERME FONKSİYONLARI ---

def generate_recommendation_prompt(genre, film_filtresi, movie_context):
    """LLM'e gönderilecek güçlendirilmiş (RAG) prompt'u oluşturur."""
    
    # TMDb verilerini prompt içine eklemek için metin formata dönüştür
    movies_string = "\n".join([
        f"Title: {m['title']}, Year: {m['year']}, Overview: {m['overview']}, Rating: {m['rating']}"
        for m in movie_context
    ])

    prompt = f"""
You are an expert AI movie critic named 'Picky Cinephile'. Your task is to provide exactly 3 personalized movie recommendations that satisfy the user's specific genre and filter requests.

**GIVEN DATA SOURCE:**
The movies listed below are sourced from The Movie Database (TMDb). You must ONLY recommend films from this list.

**USER PREFERENCES:**
Genre: {genre}
Specific Filter/Mood: {film_filtresi}

**PROVIDED MOVIE LIST (TMDb Data - Analyze the Overview, Title, and Rating to match the filter):**
---
{movies_string}
---

**RECOMMENDATION RULES:**
1.  Suggest **exactly 3 movies** from the provided list.
2.  Each suggestion must perfectly satisfy the **Genre** ({genre}) and the **Specific Filter/Mood** ({film_filtresi}).
3.  Focus the selection on movies that contain {film_filtresi} elements (e.g., strong plot twists, positive mood, emotional depth, etc.) as described in their overview.
4.  Output MUST STRICTLY follow the OUTPUT FORMAT. Do NOT include the Poster URL in the text output, we will use it separately.

**OUTPUT FORMAT:**

Movies:
1.  **[Movie Title]** ([Year]) - [Genre] | Puan: [Rating]: [Brief Explanation focusing on the Specific Filter/Mood]
2.  **[Movie Title]** ([Year]) - [Genre] | Puan: [Rating]: [Brief Explanation focusing on the Specific Filter/Mood]
3.  **[Movie Title]** ([Year]) - [Genre] | Puan: [Rating]: [Brief Explanation focusing on the Specific Filter/Mood]
"""
    return prompt

def get_gemini_recommendation(prompt):
    """Hazırlanan prompt'u Gemini'ye gönderir ve yanıtı alır."""
    if not client:
        return "Yapay zeka istemcisi başlatılamadığı için öneri alınamıyor."
        
    try:
        print("\n--- LLM'e (Picky Cinephile'a) Prompt Gönderiliyor... ---")
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt
        )
        return response.text
        
    except APIError as e:
        return f"Gemini API Hatası: Lütfen API anahtarınızı (GEMINI_API_KEY) kontrol edin. Hata: {e}"
    except Exception as e:
        return f"Genel Hata: {e}"

# --- ANA SÜREÇ (Tüm sistemin çalıştırılması) ---

# 1. TMDb'den veriyi çek
print(f"\nTMDb'den veri çekiliyor: Tür='{KULLANICI_GENRE_ADI}', Filtre='{KULLANICI_FILTRESI}'")
tmdb_data = get_movies_from_tmdb(KULLANICI_GENRE_ADI, KULLANICI_FILTRESI)

if isinstance(tmdb_data, str):
    print("\n--- Veri Çekme Başarısız ---")
    print(tmdb_data) 
elif client:
    # 2. Prompt'u oluştur (RAG: TMDb verisi prompt'a ekleniyor)
    final_prompt = generate_recommendation_prompt(KULLANICI_GENRE_ADI, KULLANICI_FILTRESI, tmdb_data)
    
    # 3. Gemini'den öneri al
    recommendation_text = get_gemini_recommendation(final_prompt)
    
    # 4. Sonucu yazdır
    print("\n==============================================")
    print(f"Picky Cinephile Önerileri | Tür: {KULLANICI_GENRE_ADI}, Filtre: {KULLANICI_FILTRESI}")
    print("==============================================")
    print(recommendation_text)
    print("\n\n--- Çekilen Verilerin Tamamı (Web Sitesi İçin Kullanılacak) ---")
    print("Bu liste, önerilen filmlerin afiş URL'lerini de içerir (Web sitesinde resimleri göstermek için):")
    
    # Bu kısım, web sitesi entegrasyonunda nasıl kullanılacağını gösterir
    for movie in tmdb_data:
        if movie['title'] in recommendation_text:
            print(f"  > {movie['title']}: {movie['poster_url']}")
            
else:
    print("\nSistem, Gemini API Anahtarı olmadan çalıştırılamadı. Lütfen .env dosyasını kontrol edin.")

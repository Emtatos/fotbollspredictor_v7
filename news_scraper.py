"""
Web scraper för att hämta aktuell matchinformation från gratis källor
- Skador från Premier Injuries
- Nyheter från BBC Sport
- Form och tabellposition från officiella källor
"""
import requests
from bs4 import BeautifulSoup
import logging
from typing import Dict, List, Optional
import re
from datetime import datetime
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FootballNewsScraper:
    """Scraper för fotbollsnyheter och skador"""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
    
    def get_premier_league_injuries(self) -> Dict[str, Dict]:
        """
        Hämtar skadedata från Premier Injuries
        
        Returns:
            Dict med lagnamn som nycklar och skadeinfo som värden
        """
        url = "https://www.premierinjuries.com/injury-table.php"
        injuries_data = {}
        
        try:
            logger.info("Hämtar skadedata från Premier Injuries...")
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Hitta alla lag och deras skador
            # Strukturen är: lagnamn följt av antal skador i en div
            team_sections = soup.find_all('div', class_=re.compile('team|injury'))
            
            # Alternativ: Parsa från breaking news-sektionen
            breaking_news = soup.find('div', class_='breaking-news')
            if breaking_news:
                news_text = breaking_news.get_text()
                
                # Extrahera information per lag
                teams = [
                    'Arsenal', 'Aston Villa', 'Bournemouth', 'Brentford', 'Brighton',
                    'Burnley', 'Chelsea', 'Crystal Palace', 'Everton', 'Fulham',
                    'Leeds', 'Leicester', 'Liverpool', 'Man City', 'Man United',
                    'Manchester City', 'Manchester United',
                    'Newcastle', 'Nottingham Forest', 'Southampton', 'Tottenham',
                    'West Ham', 'Wolves'
                ]
                
                for team in teams:
                    # Sök efter lagnamnet i texten
                    pattern = rf'{team}[:\s]+(.*?)(?={"|".join(teams)}|$)'
                    match = re.search(pattern, news_text, re.IGNORECASE | re.DOTALL)
                    
                    if match:
                        team_news = match.group(1).strip()
                        
                        # Räkna antal skador (approximation)
                        injury_keywords = ['injury', 'injured', 'out', 'sidelined', 'ruled out', 'miss']
                        injury_count = sum(1 for keyword in injury_keywords if keyword in team_news.lower())
                        
                        # Identifiera nyckelspelare
                        key_players_out = []
                        # Enkelt: hitta namn (ord med stor bokstav följt av efternamn)
                        names = re.findall(r'[A-Z][a-z]+ [A-Z][a-z]+', team_news)
                        key_players_out = names[:3]  # Max 3 namn
                        
                        injuries_data[team] = {
                            'injury_count': injury_count,
                            'key_players_out': key_players_out,
                            'news_summary': team_news[:200] + '...' if len(team_news) > 200 else team_news,
                            'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M')
                        }
            
            logger.info(f"Hämtade skadedata för {len(injuries_data)} lag")
            return injuries_data
            
        except Exception as e:
            logger.error(f"Fel vid hämtning av skadedata: {e}")
            return {}
    
    def get_championship_injuries(self) -> Dict[str, Dict]:
        """
        Hämtar skadedata för Championship från Transfermarkt
        
        Returns:
            Dict med lagnamn som nycklar och skadeinfo som värden
        """
        # Transfermarkt kräver ofta mer sofistikerad scraping
        # För nu returnerar vi tom dict och kan utöka senare
        logger.info("Championship skadedata inte implementerat än")
        return {}
    
    def get_team_news_from_bbc(self, team_name: str) -> Optional[str]:
        """
        Hämtar senaste nyheterna om ett lag från BBC Sport
        
        Args:
            team_name: Lagnamn
            
        Returns:
            Senaste nyhet eller None
        """
        try:
            # BBC Sport URL-format
            team_slug = team_name.lower().replace(' ', '-')
            url = f"https://www.bbc.com/sport/football/teams/{team_slug}"
            
            logger.info(f"Hämtar nyheter för {team_name} från BBC Sport...")
            response = self.session.get(url, timeout=10)
            
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Hitta första nyhetsrubriken
                headline = soup.find('h3', class_=re.compile('headline'))
                if headline:
                    return headline.get_text().strip()
            
            return None
            
        except Exception as e:
            logger.error(f"Fel vid hämtning av BBC-nyheter för {team_name}: {e}")
            return None
    
    def scrape_all_data(self) -> Dict:
        """
        Hämtar all tillgänglig data från alla källor
        
        Returns:
            Dict med all scrapad data
        """
        logger.info("=== STARTAR SCRAPING AV FOTBOLLSDATA ===")
        
        all_data = {
            'premier_league_injuries': self.get_premier_league_injuries(),
            'championship_injuries': self.get_championship_injuries(),
            'timestamp': datetime.now().isoformat()
        }
        
        logger.info("=== SCRAPING SLUTFÖRD ===")
        return all_data
    
    def get_injury_score(self, team_name: str, league: str = 'premier_league') -> float:
        """
        Beräknar en skadepoäng för ett lag (0-1, där 1 = många skador)
        
        Args:
            team_name: Lagnamn
            league: Liga (premier_league, championship, etc.)
            
        Returns:
            Skadepoäng mellan 0 och 1
        """
        if league == 'premier_league':
            injuries = self.get_premier_league_injuries()
        else:
            injuries = self.get_championship_injuries()
        
        if team_name in injuries:
            injury_count = injuries[team_name].get('injury_count', 0)
            # Normalisera: 0 skador = 0.0, 10+ skador = 1.0
            return min(injury_count / 10.0, 1.0)
        
        return 0.0  # Ingen data = inga kända skador


def main():
    """Testfunktion"""
    scraper = FootballNewsScraper()
    
    # Testa skadedata
    injuries = scraper.get_premier_league_injuries()
    print("\n=== PREMIER LEAGUE SKADOR ===")
    for team, data in list(injuries.items())[:5]:  # Visa första 5
        print(f"\n{team}:")
        print(f"  Skador: {data['injury_count']}")
        print(f"  Nyckelspelare ute: {', '.join(data['key_players_out'])}")
        print(f"  Sammanfattning: {data['news_summary'][:100]}...")
    
    # Testa skadepoäng
    print("\n=== SKADEPOÄNG ===")
    test_teams = ['Arsenal', 'Manchester City', 'Liverpool']
    for team in test_teams:
        score = scraper.get_injury_score(team)
        print(f"{team}: {score:.2f}")


if __name__ == "__main__":
    main()

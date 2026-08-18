# Data availability diagnosis for Stryktipset history

Utredning, ingen implementation. Ingen pipeline, ingen scraper och ingen
datainsamling har byggts. Endast enstaka verifieringsanrop har gjorts för att
fastställa struktur, räckvidd och tidsstämplar.

- Repo: `Emtatos/fotbollspredictor_v7`
- Bas: `main` @ `f3bf9c13f33d5abc238e24a384656ee39435e045`
- Läst kod: `matchday_storage.py`, `streck_import.py`, `matchday_import.py`,
  `coupon_image_parser.py`
- Verifieringsdatum: 2026-08-18

Målet på sikt är att optimera förväntad ekonomisk avkastning per kupong under
ett golv för radens träffsannolikhet. Det kräver, per omgång: pre-deadline
streck per match, rätt rad, omsättning och utdelning per vinstgrupp. Denna
rapport avgör om det finns bakåt i tiden.

---

## Fråga 1: Vad har vi redan?

### Bekräftelse: single-snapshot

Bekräftat. `save_matchday_data()` (`matchday_storage.py:203`) serialiserar
`fixtures`, `odds_by_key` och `streck_by_key` till en enda fil,
`data/saved_matchday.json` (`DEFAULT_STORAGE_PATH`), och skriver den med
`open(path, "w")`. Varje sparning ersätter den föregående. Det finns ingen
omgångsnyckel i filnamnet, ingen append, ingen versionshistorik och ingen
katalog per omgång. `clear_saved_matchday_data()` raderar filen.
`data/` är dessutom `.gitignore`:ad (rad 2: `data/`), så inget av detta är
committat.

### Finns någon annan historisk lagring?

Nej. Genomsökning av kodbasen efter `utdelning`, `payout`, `vinstgrupp`,
`omsättning`, `turnover`, `jsonl`, `sqlite`, `to_parquet` och append-mönster
mot JSON ger inga träffar som lagrar streck, rätt rad eller utdelning
historiskt. Träffarna på `history` avser uteslutande `df_history` i
feature-/Elo-koden, dvs. matchhistorik från football-data.co.uk.

Övriga datavägar in i systemet är också engångsvägar:

- `streck_import.py` läser streck från `data/streck_data.csv`
  (`DEFAULT_STRECK_PATH`), en enda fil utan omgångsfält. Kolumnerna är
  `HomeTeam, AwayTeam, Streck1, StreckX, Streck2` plus valfria `Date` och
  `Source`. Ingen tidsstämpel för när strecken observerades.
- `matchday_import.parse_streck_csv()` bygger `streck_by_key` för den aktuella
  omgången i minnet.
- `coupon_image_parser.py` tolkar en kupongskärmbild via OpenAI Vision till
  `CouponRow(home_team, away_team, streck_1/x/2, odds_1/x/2, confidence,
  notes)`. Ingen omgångsnyckel, ingen tidsstämpel, ingen persistens utanför
  `saved_matchday.json`.

På disk finns endast football-data-CSV:er (`data/E0_2425.csv` osv.),
`data/cache/` och `data/features.parquet`. Ingen streck-, rätt-rad- eller
utdelningshistorik.

### Vilka fält finns i dagens struktur?

| Nivå | Fält |
| --- | --- |
| Fil | `version`, `saved_at` (UTC, sekundupplösning), `meta.match_count`, `meta.odds_count`, `meta.streck_count` |
| `fixtures[]` | `home_team`, `away_team`, `match_key` |
| `odds_by_key[key][]` | `bookmaker`, `home`, `draw`, `away` |
| `streck_by_key[key]` | `"1"`, `"X"`, `"2"` |

### Vilka fält saknas för EV-beräkning på radnivå?

Strukturellt saknas allt utom streck och odds:

1. **Omgångsidentitet**: omgångsnummer/`drawNumber`, produktnamn, säsong/vecka.
2. **Tidsstämplar**: `regCloseTime` (spelstopp) och observationstidpunkt för
   strecken. `saved_at` säger när vi sparade, inte när strecken gällde, och
   kan lika väl ligga efter spelstopp.
3. **Utfall**: rätt rad per matchnummer (1/X/2) och resultat.
4. **Ekonomi**: omsättning (`currentNetSale`), radpris (`rowPrice`),
   utdelning och antal vinnare per vinstgrupp (13/12/11/10 rätt),
   garantifond/jackpot.
5. **Matchnummer 1–13**: dagens `match_key` är `HomeTeam_AwayTeam` utan
   kupongposition, så utdelningsberäkning per rad kan inte kopplas till
   kupongordningen.
6. **Egen insats**: vilka rader vi faktiskt lämnade in, radantal och kostnad.
7. **Historik**: alla ovanstående för mer än en omgång samtidigt. Även med
   fullständiga fält per omgång omöjliggör överskrivningen ett backtest.

Notera att Stryktipsets utdelning är pari-mutuell: EV per rad beror på hur
många andra som träffar samma antal rätt. För en EV-modell behövs därför både
`amount` och `winners` per vinstgrupp, inte bara utdelningsbeloppet.

---

## Fråga 2: Går historiska streck att hämta?

Ja, för omgångar från och med 2021-03-13, via Svenska Spels publika
draw-API. Med viktiga reservationer om pre-deadline-bevisbarhet.

### Källa 2A: `api.spela.svenskaspel.se/draw/1/stryktipset/draws/{drawNumber}`

(Samma svar ges av `api.www.svenskaspel.se/draw/1/stryktipset/draws/...`.)

Per omgång: `drawNumber`, `drawComment` (t.ex. "Stryktipset v. 2025-19"),
`drawState`, `regOpenTime`, `regCloseTime`, `rowPrice`, `currentNetSale` och
13 `drawEvents`. Per match: `eventNumber`, `eventDescription`, lag med id,
`match.matchStart`, `match.league`, `match.result` samt

```json
"svenskaFolket": {
  "one": "35", "x": "24", "two": "41", "date": "2025-05-10T16:00:02.173+02:00",
  "refOne": "36", "refX": "23", "refTwo": "41", "refDate": "2025-05-10T15:44:37.253+02:00"
}
```

Samma värden finns även strukturerat i `betMetrics.values[].distribution`
(`distribution` respektive `refDistribution`) med `distributionDate` och
`distributionRefDate`.

### Pre-deadline vs efterhandsrapporterat — avgörande fynd

Fältet finns i två varianter med **separata tidsstämplar**:

| Variant | Fält | Tidsstämpel | Status |
| --- | --- | --- | --- |
| Slutfördelning | `one`, `x`, `two` / `distribution` | `date` ≈ spelstopp + 1 sekund (t.ex. `16:00:01` mot spelstopp `15:59:00`) | Stämplad EFTER spelstopp. Speglar bara insatser lagda före spelstopp, men är inte bevisligen observerbar före deadline. |
| Referenssnapshot | `refOne`, `refX`, `refTwo` / `refDistribution` | `refDate` | Bevisligen pre-deadline **när `refDate < regCloseTime`**, vilket måste kontrolleras per omgång. |

Verifierat, per omgång (`refDate` minus `regCloseTime`):

| Omgång | Datum | Offset | Pre-deadline |
| --- | --- | ---: | --- |
| 4680 | 2021-03-13 | −14,2 min | ja |
| 4700 | 2021-07-31 | −16,1 min | ja |
| 4750 | 2022-07-16 | −16,3 min | ja |
| 4800 | 2023-06-25 | −15,0 min | ja |
| 4850 | 2024-06-01 | −14,6 min | ja |
| 4900 | 2025-05-10 | −14,4 min | ja |
| 4950 | 2026-04-25 | −1,0 min | ja |
| 4960 | 2026-07-04 | −0,2 min | ja |
| 4966 | 2026-08-15 | **+0,5 min** | **nej** |

Ref-snapshoten skiljer sig dessutom från slutvärdet (omgång 4900, match 1:
ref `36/23/41` mot slut `35/24/41`), vilket bekräftar att det är två olika
mätpunkter och inte samma tal.

Tre konsekvenser:

1. Referensvärdet är **det enda** streckvärde som kan bevisas ha varit
   observerbart före spelstopp. Slutvärdet ska behandlas som
   efterhandsrapporterat.
2. Marginalen har krympt kraftigt. 2021–2025 låg ref-snapshoten ca 14–16
   minuter före spelstopp; 2026 ligger den inom en minut och har i minst ett
   verifierat fall (4966) hamnat **efter** spelstopp. `refDate < regCloseTime`
   måste därför verifieras per omgång och omgångar som faller ska uteslutas.
3. Endast **ett** ref-snapshot bevaras per match. API:t ger ingen tidsserie
   över hur strecken rörde sig, så man kan inte välja en egen
   observationstidpunkt bakåt i tiden.

### Räckvidd

- `drawNumber` 4268 (v. 2013-03, spelstopp 2013-01-19) är det äldsta
  verifierade svaret. 4261, 4262, 4264 och 4266 ger 404, så gränsen ligger
  strax under 4268.
- 4268–4679 (2013-01 – 2021-03): `one/x/two` finns, men `date`/`refDate` är
  `0001-01-01T01:12:00+01:12` och `refOne/refX/refTwo` är `null`.
  **OANVÄNDBAR för bakåtriktad mätning** — slutfördelning utan
  tidsstämpel, dvs. efterhandsrapporterad utan möjlighet att belägga
  pre-deadline-status.
- Från 4680 (2021-03-13) finns `refOne/refX/refTwo` och en riktig `refDate`.
  Gränsen är exakt: 4676–4679 saknar dem, 4680 har dem.
- 4967 är den öppna omgången (spelstopp 2026-08-22). För öppna omgångar är
  `distributionDate` en levande tidsstämpel, dvs. samma API kan användas för
  framåtriktad insamling med valfri marginal före spelstopp.

### Bonusfält och deras begränsningar

- `startOdds` (Svenska Spels tipsodds vid öppning) finns historiskt från
  omkring 2022 (saknas i 4680 och 4700).
- `odds` (aktuella odds) är `null` i alla verifierade avslutade omgångar utom
  den senaste (4966, en vecka gammal). Stängningsodds bevaras alltså inte
  långsiktigt. Detta gör inte modellen sämre — den använder
  football-data-odds — men API:t kan inte leverera historiska stängningsodds.
- `tioTidningarsTips` finns historiskt men är `null` för den öppna omgången.

### Källa 2B: tredjepartssajter

Genomsökta: `stryketanalysen.se`, `speltjanst.se` och liknande. De publicerar
aggregerad statistik (teckenfördelning i rätt rad, omsättnings- och
utdelningssammanställningar, rätta rader per omgång) — inte per-match
1/X/2-streck med tidsstämpel. Ingen av dem skiljer på pre-deadline och
efterhandsrapporterade streck. **OANVÄNDBARA för bakåtriktad
streckmätning** i den form de publicerar.

`github.com/andreyhgl/stryktipset-results` hämtar rätt rad från samma
Svenska Spel-API (`/result`) från drawNumber 4631 och framåt. Det är
tredjeparts bekräftelse på att endpointen är publik och stabil, men tillför
ingen streckdata.

---

## Fråga 3: Går rätt rad, omsättning och utdelning att hämta?

Ja, från samma API, och längre bakåt än strecken.

### Källa 3A: `api.spela.svenskaspel.se/draw/1/stryktipset/draws/{drawNumber}/result`

Svaret innehåller:

- `events[]` med `eventNumber`, `eventDescription`, `outcome` ("1"/"X"/"2"),
  `outcomeScore.home/away`, `matchId`, `participants`, `cancelled` — dvs.
  hela rätta raden, matchnummer för matchnummer.
- `distribution[]` med **exakt de fyra vinstgrupperna**:
  `13 rätt`, `12 rätt`, `11 rätt`, `10 rätt`, var och en med `winners` och
  `amount`. Exempel, omgång 4900: 5 vinnare × 1 119 407 kr på 13 rätt,
  163 × 12 876 kr, 2 684 × 625 kr, 25 703 × 136 kr.
- `currentNetSale` (omsättning) och `regCloseTime`.

Verifierad räckvidd: fullständigt svar (13 utfall, 4 vinstgrupper,
omsättning) för 4300 (2013-08-31), 4500 (2017-07-01), 4670 (2021-01-02) och
alla nyare stickprov till 4966 (2026-08-15).

Rätt rad, omsättning och utdelning är resultatdata och kan per definition
inte vara pre-deadline; de behöver inte vara det. Ingen del av dem är
efterhandsrapporterad i problematisk mening.

Garantifondens andel och eventuell jackpot redovisas inte separat i
`distribution`; endast de fyra vinstgrupperna. Fördelningsnycklarna
(40/15/12/25 % plus 8 % garantifond) är publicerade i spelreglerna men ingår
inte i API-svaret.

---

## Fråga 4: Går källorna att koppla ihop?

Ja, trivialt, eftersom streck, rätt rad, omsättning och utdelning kommer från
**samma API och samma `drawNumber`**. Ingen heuristisk matchning mellan olika
källor behövs:

- `drawNumber` är den gemensamma omgångsnyckeln.
- `eventNumber` (1–13) kopplar streck till utfall inom omgången.
- `matchId` och `participants[].id` är stabila identifierare per match och lag.
- `match.matchStart` och `match.league` ger datum och liga.

Kopplingen mot modellens egen data (football-data.co.uk, `MatchKey` =
`HomeTeam_AwayTeam`) kräver däremot namnnormalisering via befintlig
alias-hantering, och är begränsad av liga-täckning: Stryktipsetkupongen
innehåller ofta matcher utanför E0–E3. Verifierade stickprov:

| Omgång | E0–E3-matcher av 13 | Övriga ligor på kupongen |
| --- | ---: | --- |
| 4680 (2021-03) | 12 | Svenska cupen |
| 4700 (2021-07) | 0 | Allsvenskan, Superettan, skotska Premiership, Superligaen, div. 2 |
| 4750 (2022-07) | 0 | Allsvenskan, Superettan, EM (D), Eliteserien, Veikkausliiga |
| 4800 (2023-06) | 0 | Ettan, Eliteserien, U21-EM, div. 2 |
| 4850 (2024-06) | 0 | CL, Allsvenskan, Superettan, Ettan, Eliteserien m.fl. |
| 4900 (2025-05) | 6 | Allsvenskan, Serie A, La Liga, Ligue 1, Primeira Liga |
| 4950 (2026-04) | 12 | FA Cup |
| 4966 (2026-08) | 13 | — |

Det påverkar inte datatillgången, men innebär att en EV-mätning på radnivå
antingen måste begränsas till vinterhalvårets engelska omgångar eller
kompletteras med modeller/odds för fler ligor. Detta är en modellfråga, inte
en datatillgångsfråga, och ligger utanför detta uppdrag.

### Omgångsräkning

En KOMPLETT omgång kräver: 13 matcher, pre-deadline 1/X/2-streck, rätt
slutrad, omsättning, utdelning för 13/12/11/10 rätt och en gemensam säker
omgångsnyckel.

```
verifierat minimum:      7 kompletta omgångar
möjligt maximum:       286 kompletta omgångar
okänt/ej verifierat:   279 omgångar
```

Härledning:

- **Verifierat minimum = 7**: omgångarna 4680, 4700, 4750, 4800, 4850, 4900
  och 4950 är individuellt kontrollerade och uppfyller samtliga sex krav
  (13 events, 13 pre-deadline-streck med `refDate < regCloseTime`, 13 utfall,
  4 vinstgrupper, omsättning, `drawNumber`).
- **Intervallet** för möjliga kompletta omgångar är 4680–4966, dvs. 287
  avslutade omgångar (pre-deadline-streck saknas helt före 4680; 4967 är
  ännu öppen).
- **Verifierat fallerande = 1**: omgång 4966 har `refDate` 29 sekunder efter
  spelstopp och är därmed inte komplett enligt definitionen.
- **Möjligt maximum = 287 − 1 = 286.**
- **Okänt = 279** ej individuellt kontrollerade omgångar i intervallet. Två
  osäkerheter kvarstår där: hur många som har `refDate` efter spelstopp
  (troligen främst 2026-omgångar, där marginalen krympt till under en minut)
  och om det finns hål i numreringen (404 förekommer i äldre serier, t.ex.
  4261–4266).

Att fastställa exakt antal kräver ett anrop per omgång, dvs. systematisk
nedladdning, vilket detta uppdrag uttryckligen förbjuder. Antalet kan
fastställas senare inom ett insamlingsuppdrag.

---

## Källtabell

| Källa / URL | Innehåll | Räckvidd | Pre-deadline eller efterhand | Format | Stabilitet | Documented access status / stated restrictions |
| --- | --- | --- | --- | --- | --- | --- |
| `https://api.spela.svenskaspel.se/draw/1/stryktipset/draws/{n}` (även `api.www.svenskaspel.se`) | 13 matcher, lag, liga, avspark, `svenskaFolket` (streck) + `betMetrics`, `startOdds`, `regCloseTime`, `rowPrice`, `currentNetSale` | `drawNumber` 4268 (2013-01) – 4967 (öppen). Pre-deadline-streck endast 4680 (2021-03-13) och framåt | **Båda, separerat**: `refOne/refX/refTwo` + `refDate` är pre-deadline när `refDate < regCloseTime` (verifierat 14–16 min före 2021–2025, <1 min 2026, i ett fall efter). `one/x/two` + `date` är stämplade efter spelstopp = efterhandsrapporterade | JSON, odokumenterat men stabilt schema | Officiellt Svenska Spel-API som driver deras egen webb/app. Ingen versionsgaranti; schemat kan läggas om utan förvarning. Äldre omgångar tappar fält (`odds` blir `null`) | **Oklart.** Ingen publicerad API-dokumentation eller utvecklarvillkor hittades. `robots.txt` på `www.svenskaspel.se` och `spela.svenskaspel.se` innehåller `Disallow: /api/` och `Disallow: *?draw=` (gäller de webbvärdarna). API-värden `api.spela.svenskaspel.se` svarar 500 "No route found for GET /robots.txt", dvs. saknar robots.txt. Allmänna kundvillkoren (v1.9) nämner inte automatiserad hämtning, scraping eller API-användning. Slår inte fast tillåtlighet — flaggas som oklart |
| `https://api.spela.svenskaspel.se/draw/1/stryktipset/draws/{n}/result` | Rätt rad per `eventNumber`, resultat, `distribution` med `winners` + `amount` för 13/12/11/10 rätt, `currentNetSale`, `regCloseTime` | Verifierat 4300 (2013-08) – 4966 (2026-08) | Resultatdata; pre-deadline ej tillämpligt | JSON | Samma som ovan; används av tredjepartsprojekt sedan flera år | **Oklart**, samma resonemang som ovan |
| `https://www.stryketanalysen.se/` | Rätta rader per omgång, omsättning, utdelning, aggregerad teckenstatistik | Flera år bakåt (ej verifierat i detalj) | Publicerar inte per-match-streck med tidsstämpel. **OANVÄNDBAR för bakåtriktad streckmätning** | HTML | Privat sajt, HTML kan läggas om; delar av innehållet kräver konto | `robots.txt` innehåller endast `Sitemap` och tom `User-agent: *`-regel, dvs. inga uttryckliga förbud. Villkor för databruk ej granskade → oklart |
| `https://speltjanst.se/Statistics/` | Aggregerad statistik: teckenfrekvens, omsättning, utdelningsnivåer | v. 2015-01 – v. 2020-11 för sammanställningarna | Aggregat, inga per-match-streck. **OANVÄNDBAR för bakåtriktad streckmätning** | HTML | Privat sajt | `robots.txt`: `Disallow: /lessphp/` för alla; **`Disallow: /` för SemrushBot, AhrefsBot och MJ12bot** — dvs. uttryckligt förbud mot namngivna crawlers, inte mot alla |
| `github.com/andreyhgl/stryktipset-results` | Rätta rader i CSV, hämtade från Svenska Spel-API:t | drawNumber 4631 (2020) – 4921 (2025) | Resultatdata | CSV / Python | Tredjepartsrepo, ingen drift-garanti | Publikt repo; licensvillkor ej granskade → oklart. Tillför inget utöver käll-API:t |
| Odds-aggregatorer (t.ex. `odds-api.io/sportsbooks/svenskaspel`) | Svenska Spels sportsbook-odds, inte poolspel | — | Innehåller inga Stryktipset-streck | REST | Kommersiell | Kommersiell licens krävs. Irrelevant för detta ändamål |

### Källor som uttryckligen förbjuder automatiserad hämtning

- `www.svenskaspel.se` och `spela.svenskaspel.se`: `robots.txt` har
  `Disallow: /api/`, `Disallow: /content/id/`, `Disallow: /content/filter`,
  `Disallow: /cms/documents/`, `Disallow: /cms/images/` och
  `Disallow: *?draw=`. Det utesluter att hämta streck via webbsidornas egna
  API-sökvägar eller `?draw=`-parametrade sidor på de värdarna.
- `speltjanst.se`: `Disallow: /` för SemrushBot, AhrefsBot och MJ12bot.
- Ingen källa hittades som uttryckligen förbjuder anrop mot
  `api.spela.svenskaspel.se`. Frånvaron av förbud är inte ett tillstånd:
  eftersom inga API-villkor är publicerade **flaggas tillåtligheten som
  oklar** och bör klaras ut med Svenska Spel innan systematisk hämtning görs.

---

## Rekommendation

**Kombination.**

1. **Bakåtriktad mätning är möjlig** för omgångar från 4680 (2021-03-13) och
   framåt: pre-deadline-streck (`refOne/refX/refTwo` med `refDate`), rätt
   rad, omsättning och utdelning per vinstgrupp finns i samma API med
   `drawNumber` som nyckel. Storleksordning: upp till 286 omgångar, varav 7
   är individuellt verifierade.
2. **Men bakåtriktad mätning räcker inte ensam**, av tre skäl:
   - Endast **ett** streck-snapshot per match bevaras, och dess avstånd till
     spelstopp är inte under vår kontroll. Från 2026 ligger det inom en minut
     före spelstopp och kan hamna efter, vilket gör en växande andel omgångar
     ogiltiga.
   - En strategi som mäts på ett snapshot 15 minuter före spelstopp kan bara
     spelas om man i praktiken lämnar in i det fönstret. Antagandet bör
     dokumenteras som en del av mätningen.
   - Liga-täckningen varierar från 0 till 13 av 13 E0–E3-matcher per omgång,
     så antalet omgångar som modellen faktiskt kan prissätta fullt är
     betydligt lägre än 286.
3. **Framåtriktad insamling bör starta parallellt och snarast**, eftersom den
   ger streck vid en självvald tidpunkt före spelstopp, en tidsserie i stället
   för ett snapshot och en garanterad koppling mellan uppmätt och spelbar
   information.

Nästa steg innan någon insamling byggs: klara ut tillåtligheten för
systematisk hämtning från `api.spela.svenskaspel.se` med Svenska Spel,
eftersom inga API-villkor är publicerade.

---

## Skiss: framåtriktad insamling

Ingen kod, endast skiss. Ersätter inte `saved_matchday.json` för dagens
UI-flöde utan lägger sig vid sidan om.

### Fält per omgång

Omgångsnivå: `drawNumber`, `productName`, `drawComment` (säsong/vecka),
`regOpenTime`, `regCloseTime`, `rowPrice`, `currentNetSale` vid varje
snapshot, samt `fetched_at` och en `snapshot_id`.

Matchnivå (1–13): `eventNumber`, `matchId`, hemma-/bortalag med `id` och namn,
`matchStart`, `league`, normaliserad `MatchKey`, `odds` och `startOdds`,
`svenskaFolket.one/x/two` med `date`, `refOne/refX/refTwo` med `refDate`,
`cancelled`.

Utfallsnivå (efter omgången): `outcome` per `eventNumber`, `outcomeScore`,
`distribution[].winners` och `.amount` för 13/12/11/10 rätt, slutlig
`currentNetSale`.

Egen inlämning: modellens sannolikheter vid inlämningstillfället, valda
tecken/garderingar per match, radantal, kostnad, samt vilket snapshot
beslutet fattades på. Utan detta kan realiserad EV inte utvärderas mot
förväntad.

### Lagringsstruktur

Append-only i stället för överskrivning. En katalog per omgång, en fil per
snapshot:

```
data/rounds/{drawNumber}/snapshots/{snapshot_id}.json   # tidsstämplade streck+odds
data/rounds/{drawNumber}/result.json                    # rätt rad, utdelning, omsättning
data/rounds/{drawNumber}/submission.json                # egna rader och modellsannolikheter
```

Filerna skrivs en gång och ändras aldrig; `snapshot_id` bör innehålla
observationstidpunkten. En derivatfil (Parquet/CSV) kan byggas om från
råfilerna när som helst. Katalogen ska ligga utanför `.gitignore`:ade `data/`
eller versioneras separat, annars är historiken lika förgänglig som i dag.

### Frekvens och tidpunkt

Minimum för ett giltigt backtest: **ett** snapshot med säker marginal före
spelstopp, förslagsvis T−60 min, plus **ett** direkt före spelstopp
(T−2 min) för att mäta hur mycket strecken rör sig i slutskedet. Rimlig
ambition: T−24 h, T−6 h, T−60 min, T−10 min, T−2 min, samt resultat- och
utdelningshämtning T+~6 h när omgången är `Finalized`. Det är 5–6 anrop per
omgång och vecka, dvs. försumbar last, och gör beslutstidpunkten till ett
mätbart val i stället för en okänd egenskap hos källan.

### Tid till första mätning

En Stryktipsetomgång per vecka. Med enbart framåtriktad insamling ger
- 10 omgångar ≈ 2,5 månader: räcker för att verifiera att pipelinen fångar
  rätt fält, inte för slutsatser.
- 30 omgångar ≈ 7 månader: 390 matcher, tillräckligt för första
  EV-uppskattningar med breda konfidensintervall. Utdelningsutfall på 13 rätt
  är extremt skevt, så EV-skattningen kommer domineras av 10–11-rätt-nivåerna.
- 100 omgångar ≈ 2 år: först här blir 12–13-rätt-nivåerna statistiskt
  meningsfulla.

Slutsatsen av tidsskalan är att den bakåtriktade mätningen på 2021–2025-data
bör göras även om den har svagheter — den är det enda sättet att få ett
underlag i storleksordningen 200+ omgångar inom rimlig tid — medan den
framåtriktade insamlingen startas nu för att bli det primära underlaget på
sikt.

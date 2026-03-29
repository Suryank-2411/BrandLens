"""
app.py  —  BrandLens  Streamlit UI
Calls pipeline.run_analysis() which runs the 3-node LangGraph pipeline:
  Node A → Vision (Gemini)
  Node B → Audio  (Whisper)
  Node C → Summary (Gemini — sentiment + transcript summary + insight)
"""

import streamlit as st
import json, os, time
import pandas as pd
import numpy as np
import plotly.graph_objects as pgo
from datetime import datetime

st.set_page_config(
    page_title="BrandLens — Logo Intelligence",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════════════════════
#  CSS
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');
html,body,[class*="css"]{font-family:'DM Sans',sans-serif;background-color:#080C14;color:#E8EDF5}
.stApp{background:radial-gradient(ellipse at 20% 10%,#0D1F3C 0%,#080C14 50%),
        radial-gradient(ellipse at 80% 90%,#0A1628 0%,transparent 60%);background-color:#080C14}
[data-testid="stSidebar"]{background:linear-gradient(180deg,#0D1525 0%,#080C14 100%);border-right:1px solid #1A2540}
[data-testid="stSidebar"] *{color:#B0BCCE!important}
[data-testid="stSidebar"] h1,[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3{color:#E8EDF5!important;font-family:'Syne',sans-serif!important}
#MainMenu,footer,header{visibility:hidden}
.block-container{padding-top:2rem;padding-bottom:2rem}

.hero-wrap{background:linear-gradient(135deg,#0D1F3C 0%,#091628 60%,#0A1020 100%);
  border:1px solid #1E3054;border-radius:20px;padding:3rem 3.5rem 2.5rem;
  margin-bottom:2rem;position:relative;overflow:hidden}
.hero-wrap::before{content:'';position:absolute;top:-60px;right:-60px;width:320px;height:320px;
  background:radial-gradient(circle,#1B4FD820 0%,transparent 70%);pointer-events:none}
.hero-title{font-family:'Syne',sans-serif;font-size:3rem;font-weight:800;letter-spacing:-1px;
  background:linear-gradient(135deg,#FFFFFF 0%,#7EB3FF 60%,#4A90E2 100%);
  -webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;
  margin:0 0 .4rem 0;line-height:1.1}
.hero-sub{font-size:1.05rem;color:#6B85A8;font-weight:300}
.hero-badge{display:inline-block;background:#1A3060;border:1px solid #2A4A8A;border-radius:20px;
  padding:4px 14px;font-size:.72rem;color:#7EB3FF;font-weight:600;letter-spacing:.08em;
  text-transform:uppercase;margin-bottom:1rem}

.metric-row{display:flex;gap:1rem;margin-bottom:1.5rem;flex-wrap:wrap}
.metric-card{flex:1;min-width:140px;background:linear-gradient(135deg,#0E1C33 0%,#0A1525 100%);
  border:1px solid #1A2E50;border-radius:14px;padding:1.2rem 1.4rem;position:relative;overflow:hidden}
.metric-card::after{content:'';position:absolute;top:0;left:0;right:0;height:2px;
  background:linear-gradient(90deg,#2563EB,#7EB3FF)}
.metric-val{font-family:'Syne',sans-serif;font-size:2rem;font-weight:800;color:#FFFFFF;
  line-height:1;margin-bottom:.3rem}
.metric-label{font-size:.75rem;color:#4F6A8F;text-transform:uppercase;letter-spacing:.1em;font-weight:500}

.brand-card{background:linear-gradient(135deg,#0C1A30 0%,#091422 100%);
  border:1px solid #162340;border-radius:16px;padding:1.4rem 1.6rem;
  margin-bottom:1.2rem;position:relative;overflow:hidden}
.brand-card::before{content:'';position:absolute;left:0;top:0;bottom:0;width:3px;
  background:linear-gradient(180deg,#2563EB,#7EB3FF);border-radius:2px 0 0 2px}
.brand-name{font-family:'Syne',sans-serif;font-size:1.25rem;font-weight:700;
  color:#FFFFFF;margin-bottom:.5rem}
.brand-stat{display:inline-block;background:#0D1F3C;border:1px solid #1A3060;border-radius:8px;
  padding:3px 10px;font-size:.78rem;color:#7EB3FF;margin-right:6px;margin-bottom:4px}

.appear-block{background:#0A1525;border:1px solid #132035;border-radius:12px;
  padding:1rem 1.2rem;margin-top:.8rem}
.appear-meta{display:flex;gap:1.2rem;flex-wrap:wrap;font-size:.82rem;
  color:#8AA5C8;margin-bottom:.6rem}
.appear-meta b{color:#B8CCDF}
.obj-tag{display:inline-block;background:#0F2040;border:1px solid #1A3A60;border-radius:6px;
  padding:2px 9px;font-size:.75rem;color:#7EB3FF;margin:2px 3px 4px 0}

.section-label{font-size:.7rem;text-transform:uppercase;letter-spacing:.1em;
  font-weight:700;margin:.7rem 0 .25rem}
.audio-box{border-radius:0 8px 8px 0;padding:.65rem .95rem;
  font-size:.83rem;font-style:italic;line-height:1.65;margin-bottom:.2rem}
.plain-box{border-radius:0 8px 8px 0;padding:.65rem .95rem;
  font-size:.85rem;line-height:1.7;margin-bottom:.2rem}
.summary-box{background:#071A10;border-left:3px solid #10B981;border-radius:0 8px 8px 0;
  padding:.7rem 1rem;font-size:.88rem;color:#A7F3D0;line-height:1.7}

.section-title{font-family:'Syne',sans-serif;font-size:1.3rem;font-weight:700;color:#FFFFFF;
  margin:2rem 0 1rem 0;display:flex;align-items:center;gap:.6rem}
.section-title::after{content:'';flex:1;height:1px;
  background:linear-gradient(90deg,#1A2E50,transparent);margin-left:.5rem}

.pipeline-step{display:flex;align-items:center;gap:.6rem;margin:.3rem 0;font-size:.84rem;color:#8AA5C8}
.pipeline-badge{background:#0D1F3C;border:1px solid #1A3060;border-radius:6px;
  padding:2px 9px;font-size:.72rem;font-weight:700;color:#7EB3FF;font-family:'Syne',sans-serif}

.stTextInput>div>div>input{background:#0D1A2E!important;border:1px solid #1E3054!important;
  border-radius:10px!important;color:#E8EDF5!important;font-size:.95rem!important;padding:.7rem 1rem!important}
.stTextInput>div>div>input:focus{border-color:#2563EB!important;box-shadow:0 0 0 3px #2563EB20!important}
.stTextArea textarea{background:#0D1A2E!important;border:1px solid #1E3054!important;
  border-radius:10px!important;color:#E8EDF5!important;font-size:.9rem!important}
.stTextArea textarea:focus{border-color:#2563EB!important}
.stButton>button{background:linear-gradient(135deg,#1D4ED8 0%,#2563EB 100%)!important;
  color:#FFFFFF!important;border:none!important;border-radius:10px!important;
  font-family:'Syne',sans-serif!important;font-weight:700!important;
  font-size:.95rem!important;padding:.65rem 2rem!important;width:100%!important}
[data-testid="stDownloadButton"]>button{background:linear-gradient(135deg,#064E3B,#065F46)!important;
  color:#6EE7B7!important;border:1px solid #047857!important}
[data-baseweb="tab-list"]{background:#0A1220!important;border-radius:10px!important;
  padding:4px!important;gap:4px!important;border:1px solid #162035!important}
[data-baseweb="tab"]{background:transparent!important;color:#4F6A8F!important;
  border-radius:8px!important;font-family:'Syne',sans-serif!important;
  font-weight:600!important;font-size:.85rem!important}
[aria-selected="true"][data-baseweb="tab"]{background:#1A3060!important;color:#7EB3FF!important}
[data-baseweb="select"]>div{background:#0D1A2E!important;border-color:#1E3054!important;
  border-radius:10px!important;color:#E8EDF5!important}
hr{border-color:#1A2540!important}
.empty-state{text-align:center;padding:4rem 2rem;color:#2E4060}
.empty-icon{font-size:3.5rem;margin-bottom:1rem}
.empty-title{font-family:'Syne',sans-serif;font-size:1.4rem;font-weight:700;color:#3D5470;margin-bottom:.5rem}
.empty-sub{color:#2E4060;font-size:.9rem;line-height:1.8}
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
#  CATEGORY CONFIG
# ══════════════════════════════════════════════════════════════════════════════
CATEGORY_MAP = {
    "Apple":"Tech","Samsung":"Tech","Intel":"Tech","AMD":"Tech","Nvidia":"Tech",
    "Qualcomm":"Tech","HP":"Tech","Dell":"Tech","Lenovo":"Tech","Asus":"Tech",
    "Acer":"Tech","MSI":"Tech","Sony":"Tech","LG":"Tech","Panasonic":"Tech",
    "Toshiba":"Tech","Sharp":"Tech","Philips":"Tech","Bosch":"Tech","Siemens":"Tech",
    "Huawei":"Tech","Xiaomi":"Tech","OnePlus":"Tech","Oppo":"Tech","Vivo":"Tech",
    "Realme":"Tech","Motorola":"Tech","Nokia":"Tech","HTC":"Tech","BlackBerry":"Tech",
    "Corsair":"Tech","Logitech":"Tech","Razer":"Tech","SteelSeries":"Tech",
    "Western Digital":"Tech","Seagate":"Tech","Kingston":"Tech","Sandisk":"Tech",
    "Google":"Tech","Microsoft":"Tech","Amazon":"Tech","Meta":"Tech","IBM":"Tech",
    "Oracle":"Tech","Salesforce":"Tech","SAP":"Tech","Cisco":"Tech","Adobe":"Tech",
    "Autodesk":"Tech","VMware":"Tech","Uber":"Tech","Airbnb":"Tech","Dropbox":"Tech",
    "Slack":"Tech","Zoom":"Tech","Shopify":"Tech","Stripe":"Tech","Twilio":"Tech",
    "Atlassian":"Tech","HubSpot":"Tech","Mailchimp":"Tech","GitHub":"Tech",
    "GitLab":"Tech","Docker":"Tech","Kubernetes":"Tech","Cloudflare":"Tech",
    "Discord":"Social","Reddit":"Social","Pinterest":"Social","Snapchat":"Social",
    "TikTok":"Social","Twitter":"Social","WhatsApp":"Social","Telegram":"Social",
    "Signal":"Social","Skype":"Social","WeChat":"Social","Facebook":"Social",
    "Instagram":"Social","LinkedIn":"Social",
    "YouTube":"Media","Netflix":"Media","Spotify":"Media","Disney+":"Media",
    "HBO":"Media","Hulu":"Media","Apple TV":"Media","Amazon Prime":"Media",
    "Peacock":"Media","Paramount+":"Media","Vimeo":"Media","Dailymotion":"Media",
    "SoundCloud":"Media","Deezer":"Media","Tidal":"Media","Pandora":"Media",
    "Twitch":"Media","Walt Disney":"Media","Warner Bros":"Media","Universal":"Media",
    "Paramount":"Media","Sony Pictures":"Media","20th Century":"Media",
    "Lionsgate":"Media","DreamWorks":"Media","Pixar":"Media","Marvel":"Media",
    "DC Comics":"Media","Nintendo":"Media","PlayStation":"Media","Xbox":"Media",
    "Steam":"Media","Epic Games":"Media","Activision":"Media","EA Sports":"Media",
    "Ubisoft":"Media","Rockstar Games":"Media","CNN":"Media","BBC":"Media",
    "Fox News":"Media","NBC":"Media","ABC":"Media","CBS":"Media",
    "National Geographic":"Media","Time Magazine":"Media","The New York Times":"Media",
    "Toyota":"Automotive","Honda":"Automotive","Ford":"Automotive","Chevrolet":"Automotive",
    "BMW":"Automotive","Mercedes-Benz":"Automotive","Audi":"Automotive","Volkswagen":"Automotive",
    "Tesla":"Automotive","Ferrari":"Automotive","Lamborghini":"Automotive","Porsche":"Automotive",
    "Rolls Royce":"Automotive","Bentley":"Automotive","Maserati":"Automotive","Bugatti":"Automotive",
    "McLaren":"Automotive","Aston Martin":"Automotive","Jaguar":"Automotive","Land Rover":"Automotive",
    "Volvo":"Automotive","Peugeot":"Automotive","Renault":"Automotive","Fiat":"Automotive",
    "Alfa Romeo":"Automotive","Subaru":"Automotive","Mazda":"Automotive","Mitsubishi":"Automotive",
    "Nissan":"Automotive","Hyundai":"Automotive","Kia":"Automotive","Genesis":"Automotive",
    "Lexus":"Automotive","Infiniti":"Automotive","Acura":"Automotive","Cadillac":"Automotive",
    "Lincoln":"Automotive","Jeep":"Automotive","Dodge":"Automotive","Ram":"Automotive",
    "GMC":"Automotive","Buick":"Automotive","Chrysler":"Automotive","MINI":"Automotive",
    "Smart":"Automotive","Rivian":"Automotive","Lucid":"Automotive","Polestar":"Automotive",
    "BYD":"Automotive","Harley Davidson":"Automotive","Ducati":"Automotive",
    "Yamaha":"Automotive","Kawasaki":"Automotive","Suzuki":"Automotive",
    "Nike":"Fashion","Adidas":"Fashion","Puma":"Fashion","Reebok":"Fashion",
    "Under Armour":"Fashion","New Balance":"Fashion","Converse":"Fashion","Vans":"Fashion",
    "Timberland":"Fashion","The North Face":"Fashion","Patagonia":"Fashion","Columbia":"Fashion",
    "Levi's":"Fashion","Wrangler":"Fashion","Calvin Klein":"Fashion","Tommy Hilfiger":"Fashion",
    "Ralph Lauren":"Fashion","Lacoste":"Fashion","Hugo Boss":"Fashion","Armani":"Fashion",
    "Versace":"Fashion","Gucci":"Fashion","Louis Vuitton":"Fashion","Chanel":"Fashion",
    "Hermes":"Fashion","Dior":"Fashion","Prada":"Fashion","Burberry":"Fashion",
    "Fendi":"Fashion","Givenchy":"Fashion","Balenciaga":"Fashion","Saint Laurent":"Fashion",
    "Valentino":"Fashion","Bottega Veneta":"Fashion","Zara":"Fashion","H&M":"Fashion",
    "Uniqlo":"Fashion","Gap":"Fashion","Forever 21":"Fashion","Primark":"Fashion",
    "ASOS":"Fashion","Zegna":"Fashion","Moncler":"Fashion","Stone Island":"Fashion",
    "Supreme":"Fashion","Off-White":"Fashion","Stussy":"Fashion","Champion":"Fashion",
    "Carhartt":"Fashion","Dickies":"Fashion",
    "McDonald's":"Food & Bev","KFC":"Food & Bev","Burger King":"Food & Bev",
    "Subway":"Food & Bev","Pizza Hut":"Food & Bev","Domino's":"Food & Bev",
    "Taco Bell":"Food & Bev","Wendy's":"Food & Bev","Popeyes":"Food & Bev",
    "Chick-fil-A":"Food & Bev","Five Guys":"Food & Bev","Shake Shack":"Food & Bev",
    "Starbucks":"Food & Bev","Dunkin":"Food & Bev","Tim Hortons":"Food & Bev",
    "Costa Coffee":"Food & Bev","Pret A Manger":"Food & Bev","Coca-Cola":"Food & Bev",
    "Pepsi":"Food & Bev","Sprite":"Food & Bev","Fanta":"Food & Bev","Dr Pepper":"Food & Bev",
    "Mountain Dew":"Food & Bev","Red Bull":"Food & Bev","Monster Energy":"Food & Bev",
    "Rockstar Energy":"Food & Bev","Gatorade":"Food & Bev","Powerade":"Food & Bev",
    "Nestle":"Food & Bev","Nescafe":"Food & Bev","Nespresso":"Food & Bev",
    "Lavazza":"Food & Bev","Illy":"Food & Bev","Heinz":"Food & Bev","Kelloggs":"Food & Bev",
    "Quaker":"Food & Bev","Lay's":"Food & Bev","Pringles":"Food & Bev","Doritos":"Food & Bev",
    "Oreo":"Food & Bev","Cadbury":"Food & Bev","Toblerone":"Food & Bev","Kit Kat":"Food & Bev",
    "Snickers":"Food & Bev","M&Ms":"Food & Bev","Skittles":"Food & Bev","Haribo":"Food & Bev",
    "Ferrero Rocher":"Food & Bev","Lindt":"Food & Bev","Godiva":"Food & Bev",
    "Heineken":"Food & Bev","Budweiser":"Food & Bev","Corona":"Food & Bev","Guinness":"Food & Bev",
    "Jack Daniels":"Food & Bev","Johnnie Walker":"Food & Bev","Absolut":"Food & Bev",
    "Bacardi":"Food & Bev","Smirnoff":"Food & Bev","Grey Goose":"Food & Bev",
    "Moet Chandon":"Food & Bev","Evian":"Food & Bev","Perrier":"Food & Bev",
    "San Pellegrino":"Food & Bev","Tropicana":"Food & Bev","Minute Maid":"Food & Bev",
    "Walmart":"Retail","Target":"Retail","Costco":"Retail","IKEA":"Retail","LEGO":"Retail",
    "eBay":"Retail","Etsy":"Retail","Alibaba":"Retail","AliExpress":"Retail","Flipkart":"Retail",
    "Best Buy":"Retail","Home Depot":"Retail","Sephora":"Retail","Ulta Beauty":"Retail",
    "Macy's":"Retail","Nordstrom":"Retail","Bloomingdales":"Retail","Tiffany":"Retail",
    "Cartier":"Retail","Swarovski":"Retail","Rolex":"Retail","Omega":"Retail",
    "TAG Heuer":"Retail","Casio":"Retail","Fossil":"Retail","Ray-Ban":"Retail","Oakley":"Retail",
    "FedEx":"Retail","UPS":"Retail","DHL":"Retail","USPS":"Retail","Accenture":"Retail",
    "KPMG":"Retail","Deloitte":"Retail","PwC":"Retail","EY":"Retail","McKinsey":"Retail",
    "Visa":"Finance","Mastercard":"Finance","American Express":"Finance","PayPal":"Finance",
    "Stripe":"Finance","Square":"Finance","Venmo":"Finance","Cash App":"Finance",
    "Klarna":"Finance","Revolut":"Finance","Wise":"Finance","Citibank":"Finance",
    "JPMorgan":"Finance","Goldman Sachs":"Finance","Morgan Stanley":"Finance",
    "Bank of America":"Finance","Wells Fargo":"Finance","HSBC":"Finance","Barclays":"Finance",
    "Deutsche Bank":"Finance","UBS":"Finance","Credit Suisse":"Finance","BlackRock":"Finance",
    "Fidelity":"Finance","Vanguard":"Finance","Coinbase":"Finance","Binance":"Finance",
    "ESPN":"Sports","NFL":"Sports","NBA":"Sports","FIFA":"Sports","UEFA":"Sports",
    "Olympics":"Sports","Formula 1":"Sports","NASCAR":"Sports","WWE":"Sports","UFC":"Sports",
    "Peloton":"Sports","Fitbit":"Sports","Garmin":"Sports","Polar":"Sports","GoPro":"Sports",
    "Wilson":"Sports","Head":"Sports","Callaway":"Sports","TaylorMade":"Sports",
    "Titleist":"Sports","Ping":"Sports","Yonex":"Sports","Speedo":"Sports",
    "Marriott":"Travel","Hilton":"Travel","Hyatt":"Travel","IHG":"Travel","Wyndham":"Travel",
    "Booking.com":"Travel","Expedia":"Travel","TripAdvisor":"Travel","Lyft":"Travel",
    "Emirates":"Travel","Qatar Airways":"Travel","Singapore Airlines":"Travel",
    "Lufthansa":"Travel","British Airways":"Travel","Delta":"Travel",
    "United Airlines":"Travel","American Airlines":"Travel",
    "Johnson & Johnson":"Healthcare","Pfizer":"Healthcare","Moderna":"Healthcare",
    "AstraZeneca":"Healthcare","Bayer":"Healthcare","Novartis":"Healthcare",
    "Roche":"Healthcare","Abbott":"Healthcare","Medtronic":"Healthcare",
    "Philips Healthcare":"Healthcare",
    "Shell":"Energy","BP":"Energy","ExxonMobil":"Energy","Chevron":"Energy",
    "Total":"Energy","Tesla Energy":"Energy",
    "Procter Gamble":"Household","Unilever":"Household","Colgate":"Household",
    "Oral-B":"Household","Gillette":"Household","Dove":"Household","Nivea":"Household",
    "L'Oreal":"Household","Maybelline":"Household","MAC Cosmetics":"Household",
    "Clinique":"Household","Estee Lauder":"Household","Pantene":"Household",
    "Head Shoulders":"Household","Tide":"Household","Ariel":"Household",
    "Pampers":"Household","Huggies":"Household","Kleenex":"Household",
    "Febreze":"Household","Lysol":"Household","Dettol":"Household",
}

CATEGORY_COLORS = {
    "Fashion":"#EC4899","Automotive":"#F59E0B","Food & Bev":"#EF4444",
    "Tech":"#3B82F6","Media":"#8B5CF6","Social":"#06B6D4",
    "Finance":"#10B981","Retail":"#F97316","Sports":"#84CC16",
    "Travel":"#14B8A6","Healthcare":"#F43F5E","Energy":"#EAB308",
    "Household":"#A78BFA","Other":"#6B7280",
}

# Sentiment visual config
SENTIMENT_CONFIG = {
    "PROMOTING":     ("🟢", "#10B981", "#071A10", "#6EE7B7"),
    "DEMOTING":      ("🔴", "#EF4444", "#1A0707", "#FCA5A5"),
    "NEUTRAL":       ("🟡", "#F59E0B", "#1A1200", "#FCD34D"),
    "NOT MENTIONED": ("⚪", "#6B7280", "#0D1525", "#9CA3AF"),
}

def bcolor(b): return CATEGORY_COLORS.get(CATEGORY_MAP.get(b,"Other"),"#6B7280")
def bcat(b):   return CATEGORY_MAP.get(b,"Other")

DB="#080C14"; DP="#0D1525"; GR="#1A2540"; TC="#8AA5C8"

def cdef(fig, h=420):
    fig.update_layout(height=h, paper_bgcolor=DP, plot_bgcolor=DB,
        font=dict(family="DM Sans", color=TC, size=12),
        margin=dict(l=16,r=16,t=40,b=16),
        legend=dict(bgcolor=DP, bordercolor=GR, borderwidth=1))
    fig.update_xaxes(gridcolor=GR, zerolinecolor=GR)
    fig.update_yaxes(gridcolor=GR, zerolinecolor=GR)
    return fig


# ══════════════════════════════════════════════════════════════════════════════
#  CHARTS
# ══════════════════════════════════════════════════════════════════════════════
def build_timeline(brands):
    rows=[]
    for b in brands:
        for a in b["appearances"]:
            rows.append(dict(Brand=b["brand"],Start=a["start_sec"],Dur=a["duration_sec"],
                End=a["end_sec"],Size=a["avg_size_pct"],Pos=a["avg_position"]["quadrant"],
                Color=bcolor(b["brand"]),Cat=bcat(b["brand"])))
    if not rows: return None
    df=pd.DataFrame(rows)
    order=df.groupby("Brand")["Dur"].sum().sort_values(ascending=True).index.tolist()
    fig=pgo.Figure(); seen=set()
    for _,r in df.iterrows():
        c=r["Cat"]; sl=c not in seen; seen.add(c)
        fig.add_trace(pgo.Bar(x=[r["Dur"]],y=[r["Brand"]],base=[r["Start"]],orientation="h",
            marker_color=r["Color"],marker_line_width=0,opacity=0.88,
            name=c,legendgroup=c,showlegend=sl,
            hovertemplate=f"<b>{r['Brand']}</b><br>⏱ {r['Start']:.1f}s→{r['End']:.1f}s<br>"
                          f"Dur:{r['Dur']:.1f}s  Size:{r['Size']:.1f}%<br>Pos:{r['Pos']}<extra></extra>"))
    fig.update_layout(barmode="overlay",
        yaxis=dict(categoryorder="array",categoryarray=order,tickfont=dict(size=11)),
        xaxis_title="Time (seconds)",
        title=dict(text="Brand Appearance Timeline",font=dict(family="Syne",size=15,color="#FFF")))
    return cdef(fig, max(380,len(order)*34+80))

def build_bar(brands):
    df=pd.DataFrame([dict(Brand=b["brand"],Sec=b["total_duration_sec"],
                          Color=bcolor(b["brand"])) for b in brands]).sort_values("Sec")
    fig=pgo.Figure(pgo.Bar(x=df["Sec"],y=df["Brand"],orientation="h",
        marker_color=df["Color"],marker_line_width=0,
        hovertemplate="<b>%{y}</b><br>%{x:.1f}s<extra></extra>"))
    fig.update_layout(xaxis_title="Total screen time (s)",
        title=dict(text="Total On-Screen Duration",font=dict(family="Syne",size=15,color="#FFF")))
    return cdef(fig, max(360,len(df)*32+80))

def build_heatmap(brands):
    pts=[dict(x=a["avg_position"]["x"],y=a["avg_position"]["y"],
              brand=b["brand"],dur=a["duration_sec"])
         for b in brands for a in b["appearances"]]
    if not pts: return None
    df=pd.DataFrame(pts)
    xi,yi=np.linspace(0,1,20),np.linspace(0,1,20)
    H,xe,ye=np.histogram2d(df["x"],df["y"],bins=[xi,yi],weights=df["dur"])
    fig=pgo.Figure(pgo.Heatmap(z=H.T,
        x=np.round((xe[:-1]+xe[1:])/2,2),
        y=np.round((ye[:-1]+ye[1:])/2,2),
        colorscale=[[0,DB],[0.15,"#0D1F3C"],[0.4,"#1D4ED8"],[0.7,"#3B82F6"],[1,"#93C5FD"]],
        hovertemplate="H:%{x:.2f} V:%{y:.2f}<extra></extra>",showscale=True))
    fig.add_trace(pgo.Scatter(x=df["x"],y=df["y"],mode="markers+text",
        marker=dict(size=df["dur"].clip(upper=20)*1.4+6,
                    color=[bcolor(b) for b in df["brand"]],
                    opacity=0.85,line=dict(width=1,color="rgba(255,255,255,0.19)")),
        text=df["brand"],textposition="top center",
        textfont=dict(size=9,color="#FFF"),showlegend=False,
        hovertemplate="<b>%{text}</b><br>(%{x:.2f},%{y:.2f})<extra></extra>"))
    fig.update_layout(
        xaxis=dict(title="← Left    Right →",range=[0,1],
            tickvals=[0,.25,.5,.75,1],ticktext=["Left","¼","Centre","¾","Right"]),
        yaxis=dict(title="↑ Top    Bottom ↓",range=[0,1],
            tickvals=[0,.25,.5,.75,1],ticktext=["Top","¼","Centre","¾","Bottom"]),
        title=dict(text="Logo Position Heatmap",font=dict(family="Syne",size=15,color="#FFF")))
    return cdef(fig, 480)

def build_pie(brands):
    ct={}
    for b in brands: ct[bcat(b["brand"])]=ct.get(bcat(b["brand"]),0)+b["total_duration_sec"]
    labels=list(ct.keys()); values=list(ct.values())
    fig=pgo.Figure(pgo.Pie(labels=labels,values=values,hole=0.55,
        marker=dict(colors=[CATEGORY_COLORS.get(l,"#6B7280") for l in labels],
                    line=dict(color=DB,width=2)),
        textinfo="label+percent",textfont=dict(size=11,color="#FFF"),
        hovertemplate="<b>%{label}</b><br>%{value:.1f}s<extra></extra>"))
    fig.update_layout(title=dict(text="Screen Time by Category",
        font=dict(family="Syne",size=15,color="#FFF")),showlegend=False)
    return cdef(fig, 380)


# ══════════════════════════════════════════════════════════════════════════════
#  EXPORT
# ══════════════════════════════════════════════════════════════════════════════
def to_csv(brands):
    rows=[]
    for b in brands:
        for i,a in enumerate(b["appearances"],1):
            p=a["avg_position"]
            rows.append({
                "Brand":                   b["brand"],
                "Category":                bcat(b["brand"]),
                "Appearance#":             i,
                "Start(s)":                a["start_sec"],
                "End(s)":                  a["end_sec"],
                "Duration(s)":             a["duration_sec"],
                "Size%":                   a["avg_size_pct"],
                "X":                       p["x"],
                "Y":                       p["y"],
                "Quadrant":                p["quadrant"],
                "Confidence":              a["best_confidence"],
                "Objects":                 ", ".join(a.get("objects",[])),
                "AudioContext(+-10s)":     a.get("audio_context",""),
                "AudioSentiment":          a.get("audio_sentiment",""),
                "HowPromotedDemoted":      a.get("audio_how",""),
                "TranscriptSummary":       a.get("audio_transcript_summary",""),
                "AIInsight":               a.get("summary",""),
            })
    return pd.DataFrame(rows).to_csv(index=False).encode()

def to_json(brands, url):
    return json.dumps({
        "analysed_at":  datetime.now().isoformat(),
        "video_url":    url,
        "total_brands": len(brands),
        "brands":       brands,
    }, indent=2).encode()


# ══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("### 🔑 Gemini API Key")
    gemini_key = st.text_input("Key",
        value=os.environ.get("GEMINI_API_KEY",""),
        type="password", label_visibility="collapsed",
        help="Free key at https://aistudio.google.com")
    if not gemini_key:
        st.warning("Enter your Gemini API key to run.")

    st.markdown("---")
    st.markdown("### ⚙️ Settings")
    sample_fps = st.slider("Frames per second", 1, 3, 1,
        help="Higher = more accurate but more API calls.")
    conf_min   = st.slider("Min confidence", 0.3, 0.9, 0.5, 0.05,
        help="Drop detections below this score.")

    st.markdown("---")
    st.markdown("### 🤖 Current Model")
    st.code("gemini-1.5-flash\n(demo / free tier)", language=None)
    st.caption("Change `GEMINI_MODEL` in pipeline.py to upgrade.")

    st.markdown("---")
    st.markdown("### 🔄 Pipeline")
    for badge, label in [
        ("A", "Vision — logo + context"),
        ("B", "Audio — Whisper transcript"),
        ("C", "Summary — sentiment + insight"),
    ]:
        st.markdown(
            f'<div class="pipeline-step">'
            f'<span class="pipeline-badge">{badge}</span>{label}</div>',
            unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 📊 Sentiment Key")
    for s,(icon,bc,_,tc) in SENTIMENT_CONFIG.items():
        st.markdown(
            f'<div style="display:flex;align-items:center;gap:8px;margin:3px 0">'
            f'<span>{icon}</span>'
            f'<span style="font-size:.8rem;color:{tc}">{s}</span></div>',
            unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 🎨 Categories")
    for cat, color in CATEGORY_COLORS.items():
        st.markdown(
            f'<div style="display:flex;align-items:center;gap:8px;margin:3px 0">'
            f'<div style="width:10px;height:10px;border-radius:2px;background:{color}"></div>'
            f'<span style="font-size:.8rem;color:#8AA5C8">{cat}</span></div>',
            unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  SESSION STATE
# ══════════════════════════════════════════════════════════════════════════════
for k,v in [("results",None),("last_url",""),("last_prompt","")]:
    if k not in st.session_state: st.session_state[k]=v


# ══════════════════════════════════════════════════════════════════════════════
#  HERO
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="hero-wrap">
  <div class="hero-badge">LangGraph · Gemini Vision · Whisper Audio · Sentiment Analysis</div>
  <div class="hero-title">BrandLens</div>
  <div class="hero-sub">
    Paste a YouTube URL or S3 link → detect brand logos with context,
    transcribe ±10s of audio per appearance, analyse brand sentiment,
    and get an AI-generated insight for every detection.
  </div>
</div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  INPUT
# ══════════════════════════════════════════════════════════════════════════════
col_url, col_btn = st.columns([5,1])
with col_url:
    url_input = st.text_input("url",
        placeholder="YouTube URL  /  s3://bucket/video.mp4  /  https://bucket.s3.amazonaws.com/video.mp4",
        label_visibility="collapsed")
with col_btn:
    go = st.button("🔍  Analyse", use_container_width=True)

prompt_input = st.text_area(
    "Optional instructions for the AI",
    placeholder='e.g. "Focus on logos on clothing and sports equipment" or "Flag any Nike competitor brands"',
    height=75,
    label_visibility="visible",
)


# ══════════════════════════════════════════════════════════════════════════════
#  RUN PIPELINE
# ══════════════════════════════════════════════════════════════════════════════
if go:
    if not url_input.strip():
        st.warning("Please enter a video URL.")
    elif not gemini_key.strip():
        st.warning("Please enter your Gemini API key in the sidebar.")
    else:
        st.session_state.results     = None
        st.session_state.last_url    = url_input.strip()
        st.session_state.last_prompt = prompt_input.strip()

        status_el  = st.empty()
        prog_el    = st.progress(0, text="Starting pipeline…")
        logs: list = []

        def on_progress(msg: str):
            logs.append(msg)
            # Estimate progress
            pct = 5
            if   "Node A" in msg and "%" in msg:
                try: pct = 10 + int(msg.split("[")[-1].replace("%]","")) // 2
                except: pct = 30
            elif "Node A ✅" in msg: pct = 60
            elif "Node B"    in msg: pct = 65
            elif "Node B ✅" in msg: pct = 75
            elif "Node C"    in msg: pct = 80
            elif "Node C ✅" in msg: pct = 98
            prog_el.progress(min(pct, 98), text=msg[:90])

            done_keywords = ["✅","complete","done","ready"]
            status_el.markdown(
                '<div style="background:#0A1828;border:1px solid #1A2E50;border-radius:12px;'
                'padding:.9rem 1.3rem;font-family:monospace;font-size:.82rem;line-height:1.9">'
                + "".join(
                    f'<div style="color:{"#6EE7B7" if any(w in m for w in done_keywords) else "#7EB3FF"}">'
                    f'{"✅" if any(w in m for w in done_keywords) else "⟳"} {m}</div>'
                    for m in logs[-14:]
                ) + "</div>", unsafe_allow_html=True)

        try:
            import pipeline_v3 as pl
            results = pl.run_analysis(
                video_source   = url_input.strip(),
                gemini_api_key = gemini_key.strip(),
                user_prompt    = prompt_input.strip(),
                sample_fps     = sample_fps,
                confidence_min = conf_min,
                progress_cb    = on_progress,
            )
            prog_el.progress(100, text="Analysis complete! ✅")
            time.sleep(0.6)
            prog_el.empty()
            status_el.empty()
            st.session_state.results = results
            st.rerun()

        except Exception as e:
            prog_el.empty()
            st.error(f"❌ Pipeline error: {e}")


# ══════════════════════════════════════════════════════════════════════════════
#  RESULTS
# ══════════════════════════════════════════════════════════════════════════════
if st.session_state.results is not None:
    brands = st.session_state.results
    url    = st.session_state.last_url

    if not brands:
        st.markdown("""<div class="empty-state"><div class="empty-icon">🔎</div>
          <div class="empty-title">No brand logos detected</div>
          <div class="empty-sub">Try lowering the confidence threshold in the sidebar,<br>
          or test with a video that has clearly visible brand logos.</div>
          </div>""", unsafe_allow_html=True)
    else:
        tt   = sum(b["total_duration_sec"] for b in brands)
        ta   = sum(b["appearance_count"]   for b in brands)
        top  = brands[0]["brand"]
        cats = len(set(bcat(b["brand"]) for b in brands))

        # Sentiment summary counts
        all_sentiments = [
            a.get("audio_sentiment","")
            for b in brands for a in b["appearances"]
        ]
        n_promoting = all_sentiments.count("PROMOTING")
        n_demoting  = all_sentiments.count("DEMOTING")

        st.markdown(f"""<div class="metric-row">
          <div class="metric-card">
            <div class="metric-val">{len(brands)}</div>
            <div class="metric-label">Brands Detected</div></div>
          <div class="metric-card">
            <div class="metric-val">{tt:.0f}s</div>
            <div class="metric-label">Total Logo Time</div></div>
          <div class="metric-card">
            <div class="metric-val">{ta}</div>
            <div class="metric-label">Appearances</div></div>
          <div class="metric-card">
            <div class="metric-val" style="color:#6EE7B7">{n_promoting}</div>
            <div class="metric-label">🟢 Promoting</div></div>
          <div class="metric-card">
            <div class="metric-val" style="color:#FCA5A5">{n_demoting}</div>
            <div class="metric-label">🔴 Demoting</div></div>
          <div class="metric-card">
            <div class="metric-val" style="font-size:1.1rem">{top}</div>
            <div class="metric-label">Most Visible Brand</div></div>
        </div>""", unsafe_allow_html=True)

        t1, t2, t3, t4 = st.tabs([
            "📊  Timeline", "🏷  Brand Insights", "📍  Heatmap", "📥  Export"
        ])

        # ── Tab 1: Timeline ───────────────────────────────────────────────────
        with t1:
            st.markdown('<div class="section-title">⏱ When Did Each Brand Appear?</div>',
                        unsafe_allow_html=True)
            c1, c2 = st.columns([3,2])
            with c1:
                f = build_timeline(brands)
                if f: st.plotly_chart(f, use_container_width=True)
            with c2:
                f = build_pie(brands)
                if f: st.plotly_chart(f, use_container_width=True)
            st.markdown('<div class="section-title">📏 Total Screen Time</div>',
                        unsafe_allow_html=True)
            f = build_bar(brands)
            if f: st.plotly_chart(f, use_container_width=True)

        # ── Tab 2: Brand Insights ─────────────────────────────────────────────
        with t2:
            st.markdown('<div class="section-title">🏷 Brand Placement Insights</div>',
                        unsafe_allow_html=True)
            cats_list = sorted(set(bcat(b["brand"]) for b in brands))
            sel = st.selectbox("Filter by category", ["All"] + cats_list)
            filtered = brands if sel == "All" else [
                b for b in brands if bcat(b["brand"]) == sel
            ]

            for b in filtered:
                color  = bcolor(b["brand"])
                cat    = bcat(b["brand"])
                apps   = b["appearance_count"]
                dur    = b["total_duration_sec"]
                avg_sz = sum(a["avg_size_pct"] for a in b["appearances"]) / apps

                # Sort appearances by confidence descending
                sorted_apps = sorted(b["appearances"],
                                     key=lambda a: a["best_confidence"], reverse=True)
                appear_html = ""

                for i, a in enumerate(sorted_apps, 1):
                    p         = a["avg_position"]
                    objs      = a.get("objects", [])
                    audio     = a.get("audio_context", "")
                    sentiment = a.get("audio_sentiment", "")
                    how       = a.get("audio_how", "")
                    ts_summ   = a.get("audio_transcript_summary", "")
                    summ      = a.get("summary", "")

                    # Object tags
                    obj_tags = "".join(
                        f'<span class="obj-tag">📦 {o}</span>' for o in objs
                    ) if objs else '<span class="obj-tag">📦 unknown object</span>'

                    # Sentiment badge
                    icon, bc, bgc, txc = SENTIMENT_CONFIG.get(
                        sentiment, ("⚪","#6B7280","#0D1525","#9CA3AF"))
                    sentiment_badge = (
                        f'<span style="background:{bgc};border:1px solid {bc};'
                        f'border-radius:6px;padding:2px 10px;font-size:.76rem;'
                        f'font-weight:700;color:{txc};margin-left:8px">'
                        f'{icon} {sentiment}</span>'
                    ) if sentiment else ""

                    # Audio section — only if audio exists
                    audio_section = ""
                    if audio.strip():
                        audio_section = f"""
                        <div class="section-label" style="color:#6366F1">
                          🎙 Audio Transcript (±10s)
                        </div>
                        <div class="audio-box"
                             style="background:#07101F;border-left:3px solid #4F46E5;
                                    color:#A5B4FC">
                          "{audio}"
                        </div>

                        <div class="section-label" style="color:{bc};margin-top:.6rem">
                          📊 Brand Sentiment {sentiment_badge}
                        </div>
                        <div class="plain-box"
                             style="background:{bgc};border-left:3px solid {bc};color:{txc}">
                          {how or "—"}
                        </div>

                        <div class="section-label" style="color:#94A3B8;margin-top:.6rem">
                          🗣 What's Being Said
                        </div>
                        <div class="audio-box"
                             style="background:#07101F;border-left:3px solid #475569;
                                    color:#94A3B8;font-style:normal">
                          {ts_summ or "—"}
                        </div>
                        """

                    # AI insight
                    insight_section = ""
                    if summ:
                        insight_section = f"""
                        <div class="section-label" style="color:#10B981;margin-top:.6rem">
                          ✨ AI Insight
                        </div>
                        <div class="summary-box">{summ}</div>
                        """

                    appear_html += f"""
                    <div class="appear-block">
                      <div class="appear-meta">
                        <b>#{i}</b>
                        <span>⏱ {a["start_sec"]:.1f}s → {a["end_sec"]:.1f}s</span>
                        <span>🕐 {a["duration_sec"]:.1f}s</span>
                        <span>📐 {a["avg_size_pct"]:.1f}% of screen</span>
                        <span>📍 {p["quadrant"]} ({p["x"]:.2f}, {p["y"]:.2f})</span>
                        <span>🎯 conf {a["best_confidence"]:.3f}</span>
                      </div>
                      <div style="margin-bottom:.5rem">{obj_tags}</div>
                      {audio_section}
                      {insight_section}
                    </div>"""

                st.markdown(f"""
                <div class="brand-card">
                  <div style="display:flex;align-items:center;
                              justify-content:space-between;margin-bottom:.6rem">
                    <div class="brand-name">{b["brand"]}</div>
                    <div style="font-size:.75rem;color:{color};background:{color}18;
                                border:1px solid {color}40;border-radius:6px;
                                padding:2px 10px">{cat}</div>
                  </div>
                  <span class="brand-stat">⏱ {dur:.1f}s total</span>
                  <span class="brand-stat">👁 {apps} appearance{"s" if apps>1 else ""}</span>
                  <span class="brand-stat">📐 avg {avg_sz:.1f}% screen</span>
                  {appear_html}
                </div>""", unsafe_allow_html=True)

        # ── Tab 3: Heatmap ────────────────────────────────────────────────────
        with t3:
            st.markdown('<div class="section-title">📍 Where on Screen?</div>',
                        unsafe_allow_html=True)
            st.caption("Brighter = more logo time. Bubble size proportional to duration.")
            f = build_heatmap(brands)
            if f: st.plotly_chart(f, use_container_width=True)
            else: st.info("Not enough position data to render heatmap.")

        # ── Tab 4: Export ─────────────────────────────────────────────────────
        with t4:
            st.markdown('<div class="section-title">📥 Download Results</div>',
                        unsafe_allow_html=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**CSV Spreadsheet**")
                st.caption("Includes objects, audio transcript, sentiment, transcript summary & AI insight.")
                st.download_button("⬇️  Download CSV", to_csv(brands),
                    f"brandlens_{ts}.csv", "text/csv", use_container_width=True)
            with c2:
                st.markdown("**JSON Report**")
                st.caption("Full structured data with all fields.")
                st.download_button("⬇️  Download JSON", to_json(brands, url),
                    f"brandlens_{ts}.json", "application/json", use_container_width=True)
            st.markdown("---")
            dfp = pd.DataFrame([{
                "Brand":         b["brand"],
                "Category":      bcat(b["brand"]),
                "Total Time(s)": b["total_duration_sec"],
                "Appearances":   b["appearance_count"],
                "Avg Size(%)":   round(sum(a["avg_size_pct"] for a in b["appearances"]) / b["appearance_count"], 2),
                "Top Sentiment": max(
                    (a.get("audio_sentiment","") for a in b["appearances"]),
                    key=lambda s: ["PROMOTING","DEMOTING","NEUTRAL","NOT MENTIONED",""].index(s)
                    if s in ["PROMOTING","DEMOTING","NEUTRAL","NOT MENTIONED",""] else 99,
                    default="—"
                ),
                "Main Position": b["appearances"][0]["avg_position"]["quadrant"],
            } for b in brands])
            st.dataframe(dfp, use_container_width=True, hide_index=True)

else:
    if not go:
        st.markdown("""<div class="empty-state">
          <div class="empty-icon">🎬</div>
          <div class="empty-title">Ready to analyse</div>
          <div class="empty-sub">
            Paste a YouTube URL or S3 link above.<br>
            Add your Gemini API key in the sidebar.<br>
            Optionally add extra instructions, then click Analyse.<br><br>
            <b style="color:#3D5470">
              Pipeline: Vision (logos + context) → Audio (Whisper) → Summary (sentiment + insight)
            </b>
          </div></div>""", unsafe_allow_html=True)

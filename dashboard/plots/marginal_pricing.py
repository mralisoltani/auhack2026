from src.data_loader import load_generation_pivot
from src.data_loader import load_spot_price 
import streamlit as st
import matplotlib.pyplot as plt

def evaluate_renewable_business_case(zone: str, start: str, end: str):
  
    try:
        gen_pivot = load_generation_pivot(zone)
        price_df = load_spot_price(zone)
    except FileNotFoundError:
        st.error(f"Missing data for {zone}")
        return

    df = gen_pivot.join(price_df, how="inner").loc[start:end]
    
    renew_keywords = [
        'SOLAR', 'WIND-OFFSHORE', 'WIND-ONSHORE', 'BIOMASS', 
        'GEOTHERMAL', 'HYDRO-ROR', 'HYDRO-WATER-RESERVOIR', 
        'OTHER-RENEWABLE', 'WASTE'
    ]
    
    renew_cols = [c for c in df.columns if any(k in c.upper() for k in renew_keywords)]
    
    if not renew_cols:
        st.warning(f"No renewable generation types found in the columns for {zone}.")
        return

    df['total_renew_mw'] = df[renew_cols].sum(axis=1)
    df['rent_eur'] = df['total_renew_mw'] * df['price']
    
    total_rent = df['rent_eur'].sum()
    total_gen = df['total_renew_mw'].sum()
    avg_price = df['price'].mean()
    capture_price = total_rent / total_gen if total_gen > 0 else 0
    capture_factor = capture_price / avg_price if avg_price != 0 else 0
    st.header(f"Business Case: {zone}")   
    m1, m2, m3 = st.columns(3)
    m1.metric("Market Avg Price", f"€{avg_price:.2f}")
    m2.metric("Renewable Capture Price", f"€{capture_price:.2f}")
    m3.metric("Capture Factor", f"{capture_factor:.2%}")
    
    fig, ax = plt.subplots(figsize=(10, 4))
    
    ax.fill_between(df.index, 0, df['price'], 
                    where=(df['price'] > 0),
                    alpha=0.3, label="Inframarginal Rent (Profit)")
    
    ax.plot(df.index, df['price'], lw=1.5, label="Market Clearing Price")
    ax.set_ylabel("Euro / MWh")
    ax.set_title(f"The Marginal Pricing 'Profit Gap' in {zone}")
    ax.legend()
    ax.grid(axis='y', alpha=0.2)
    st.pyplot(fig)
    plt.close(fig) 
    st.subheader("The Inframarginal Profit Logic")
    st.write(f"""
    Because of **Marginal Pricing**, renewable sources in **{zone}** don't just get paid their costs; 
    they are paid the price of the most expensive unit needed. 
    
    - **Total 'Free' Revenue:** €{total_rent:,.2f} 
    - **Business Edge:** Renewables earned **€{capture_price:.2f}/MWh** despite having near-zero fuel costs.
    """)
    
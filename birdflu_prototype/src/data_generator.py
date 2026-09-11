import numpy as np
import pandas as pd
import random
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "data" / "synthetic_data.csv"

def gen_one(idx):
    age = int(np.clip(np.random.normal(40, 18), 0, 100))
    gender = random.choice(['M', 'F', 'Other'])
    occupation = random.choice(['Farmer', 'Healthcare', 'Student', 'Office', 'Other'])
    travel = random.choice([0, 0, 0, 1])
    contact_birds = random.choice([0, 0, 1])
    poultry_visit = random.choice([0, 0, 1])
    undercooked = random.choice([0, 0, 0, 1])
    contact_person = random.choice([0, 0, 0, 1])

    # Base infection probability
    p_infected = 0.08
    if contact_birds or poultry_visit:
        p_infected = 0.20
    if contact_birds and poultry_visit:
        p_infected = 0.30
    if contact_person:
        p_infected += 0.10
    if travel:
        p_infected += 0.05
    if occupation == 'Farmer':
        p_infected += 0.05
    if occupation == 'Healthcare':
        p_infected += 0.05
    p_infected = min(p_infected, 0.95)

    infected = np.random.rand() < p_infected

    def symp(p_base):
        p = p_base + (0.40 if infected else 0)
        return 1 if np.random.rand() < min(p, 0.98) else 0

    fever       = symp(0.20)
    cough       = symp(0.25)
    sore_throat = symp(0.12)
    muscle_ache = symp(0.18)
    headache    = symp(0.18)
    malaise     = symp(0.22)
    runny_nose  = symp(0.18)
    diarrhea    = symp(0.06)
    dyspnea     = symp(0.04)

    # Lab values — abnormal when infected
    # Infected patients get low WBC (leukopenia is a hallmark of bird flu)
    wbc_mean = 7000 - (2500 if infected else 0)
    wbc = int(np.clip(np.random.normal(wbc_mean, 1200), 2000, 20000))
    pcr = 1 if (np.random.rand() < (0.88 if infected else 0.01)) else 0
    chest_xray = 1 if (np.random.rand() < (0.55 if infected else 0.04)) else 0

    # Re-evaluate: if PCR positive AND contact with birds, always treat as infected
    if pcr == 1 and (contact_birds or poultry_visit):
        infected = True

    return {
        'id': idx, 'age': age, 'gender': gender, 'occupation': occupation,
        'travel': travel, 'contact_birds': contact_birds, 'poultry_visit': poultry_visit,
        'undercooked': undercooked, 'contact_person': contact_person,
        'fever': fever, 'cough': cough, 'sore_throat': sore_throat,
        'muscle_ache': muscle_ache, 'headache': headache, 'malaise': malaise,
        'runny_nose': runny_nose, 'diarrhea': diarrhea, 'dyspnea': dyspnea,
        'wbc': wbc, 'rt_pcr': pcr, 'chest_xray': chest_xray,
        'infected': int(infected)
    }

def generate(n=10000, seed=42):
    np.random.seed(seed)
    random.seed(seed)
    rows = [gen_one(i) for i in range(n)]
    df = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    infected_count = df['infected'].sum()
    print(f'Saved {n} rows to {OUT}')
    print(f'Infected: {infected_count} ({infected_count/n*100:.1f}%)  |  Healthy: {n - infected_count}')
    return df

if __name__ == '__main__':
    generate(10000)

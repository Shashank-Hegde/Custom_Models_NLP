"""
faiss_index_build.py

This script builds a candidate dictionary from fixed lists of symptom names (and synonyms), medications,
and locations. It encodes the candidate phrases using an SBERT model and builds a FAISS index for
fast cosine similarity search.

Before running, ensure:
    pip install faiss-cpu numpy sentence-transformers spacy nltk

Note: If you encounter numpy dtype incompatibility errors, reinstall numpy and dependent libraries
(e.g., pip install --force-reinstall numpy==1.21.6).
"""

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import json

# ----------------------------
# Define candidate lists
# ----------------------------
symptom_list = [
'fever', 'cold', 'runny nose', 'sneezing', 'rash', 'back spasm', 'dizziness', 'weakness', 'loss of appetite', 'cough', 'muscle pain', 'joint pain',
'chest pain', 'back pain', 'constipation', 'throat pain', 'diarrhea', 'flu', 'shortness of breath', 'rapid breathing', 'stomach pain', 'migraine',
'skin burning', 'itching', 'swelling', 'vomiting', 'infection', 'inflammation', 'cramp', 'bleeding', 'irritation', 'anxiety', 'depression','congestion',
'nausea', 'swollen lymph nodes', 'insomnia', 'cancer', 'diabetes', 'allergy', 'weight loss', 'weight gain', 'hair loss', 'blurred vision', 'ear pain',
'numbness', 'dry mouth', 'frequent urination', 'acne', 'confusion', 'memory loss', 'difficulty swallowing', 'restlessness', 'bloating',
'gas', 'indigestion', 'heartburn', 'mouth sore', 'nosebleed', 'ear ringing', 'dark urine', 'blood in urine', 'blood in stool', 'high blood pressure',
'low blood pressure', 'excessive thirst', 'dehydration', 'skin burning', 'sweat', 'eye pain',  'eye discharge', 'ear discharge', 'jaundice',
'hearing loss', 'balance problem', 'irregular heartbeat', 'fainting', 'tremor', 'nervousness', 'panic attack', 'mood swing', 'difficulty concentrating',
'hallucination', 'lack of motivation', 'exhaustion', 'bone pain', 'wrist pain', 'sprain', 'strain', 'arthritis', 'gout', 'headache', 'injury', 'chills','mouth pain'
'leg pain', 'hand pain', 'arm pain', 'foot pain', 'knee pain', 'shoulder pain', 'hip pain', 'jaw pain', 'tooth pain','sleepy', 'bone fracture','sleepy','back bone issue',
'female issue', 'thyroid', 'piles', 'asthma','pneumonia','sugar',
# weakness symtom
'eye weakness','leg weakness'
  #'yellow eyes', 'red eyes'
]

symptom_synonyms = {
  'back spasm': [
        'back is spasming', 'back spasms', 'back spasm', 'spinal contraction', 'muscle cramp in back', 'tight back muscles', 'back tightening', 'muscle spasm in lower back', 'spine spasming',
        'back muscle jerk', 'severe back cramp', 'spine knot', 'muscle twitch in back', 'spinal spasm', 'back stiffening', 'lower back stiffness', 'muscle contraction in back', 'back cramping',
        'muscle twitching in spine', 'sharp back pain', 'sudden back pain', 'painful muscle contraction', 'sharp spasms in back', 'throbbing back pain', 'spine twinge', 'muscle jerking in back',
        'painful back convulsion', 'tense back', 'tightness in back', 'cramping back muscles', 'pulled back muscle', 'twisting back pain', 'cramped spine', 'jerking back muscles', 'painful back tightening',
        'spinal muscle strain', 'back tension', 'muscle knots in back', 'spine tightening pain', 'painful back stiffness', 'spinal muscles seizing', 'back spasm attack', 'muscle discomfort in back',
        'intense back strain', 'stiffened spine', 'spinal muscles hardening', 'acute back spasm', 'back contorting', 'back stiffness attack', 'back muscle tension', 'back pain with spasms'
    ],
    'headache': [
        'head pain', 'throbbing headache', 'pounding head', 'splitting headache', 'severe headache', 'migraine-like ache', 'cranial ache', 'head pressure', 'sinus headache', 'tension headache',
        'hammering pain in skull', 'aching brain', 'full-head ache', 'temple-throbbing pain', 'dull ache behind eyes', 'stabbing head sensation', 'skull-crushing pressure', 'nagging ache in head',
        'relentless cranial pounding', 'forehead-tightening discomfort', 'vice-like grip on head', 'pulsating headache', 'dull throb', 'piercing head agony', 'continuous headache hum', 'low-level head strain',
        'top-of-head soreness', 'subcranial ache', 'stabbing darts of pain in scalp', 'brain-squeezing feeling', 'top-heavy ache', 'ear-to-ear head ache', 'all-encompassing head discomfort', 'band-like pressure around head',
        'persistent noggin ache', 'head tenderness', 'scalp-aching feeling', 'sensitive head region', 'brainache', 'mind-throbbing torment', 'front-lobe pressure', 'crown-of-head tension',
        'behind-the-eyes ache', 'skull-tight discomfort', 'never-ending head throb', 'grating ache inside skull', 'sinus-pressured ache', 'temple pounding', 'brain pulsation pain', 'cephalic torment',
        'oppressive ache under cranium', 'subtle persistent ache', 'gnawing head discomfort', 'dull pounding drumbeat in head', 'hammering inside skull walls', 'unyielding head tension', 'rote ache cycling through head',
        'cranium under siege', 'deep-set head pang', 'swirling headache sensation', 'anchor-like pressure in head'
    ],
    'migraine': [
        'intense one-sided headache', 'migraine aura', 'pulsating pain in head', 'photophobia-associated headache', 'debilitating headache', 'migraine attack', 'searing half-skull ache', 'throbbing temple migraine',
        'nausea-laced head pain', 'light-sensitive head torture', 'migraine episode', 'crippling one-sided ache', 'sharp lancing head pain', 'skull-splitting half-side ache', 'throbbing migraine pulse', 'debilitating cranial assault',
        'severe sensory headache', 'disabling one-sided throb', 'catastrophic temple pounding', 'migraine-induced nausea', 'half-head agony', 'sharp lancing head pain', 'stabbing head sensation', 'pulsating migraine',
        'tension-triggered migraine', 'blinding headache', 'brain-splitting side ache', 'overwhelming migraine pressure', 'incapacitating headache event', 'shattering unilateral head pain', 'sensitive to slightest sound',
        'migraine meltdown', 'severe sensitivity headache', 'hammering half-head ache', 'aura shimmer leading to pain', 'throbbing unilateral agony', 'needle-like head stab', 'crushing half-skull sensation',
        'crippling light-triggered pain', 'migraine climate inside head', 'tidal wave of head torment', 'migraine crescendo', 'migraine-flare crisis', 'incapacitating halo of pain', 'ear-to-temple throbbing on one side'
    ],
    'allergy': [
        'allergies', 'allergic reaction', 'allergic response', 'hay fever', 'allergic rhinitis', 'pollen sensitivity', 'dust mite allergy', 'food allergy', 'skin allergy', 'seasonal allergies',
        'environmental allergies', 'allergic condition', 'allergic response to pollen', 'sensitive to allergens', 'sneezing due to allergies', 'wheezing from allergic reaction',
        'swollen nasal passages', 'runny nose from allergies', 'sinus congestion from allergies', 'allergic rashes', 'eczema flare-up', 'hives', 'itchy skin from allergens', 'swollen face from allergies',
        'respiratory allergy', 'allergic reactions in skin', 'excessive histamine release', 'redness from allergy', 'swollen throat from allergies', 'asthma attack triggered by allergens', 'increased mucus production',
        'throat irritation due to allergens', 'difficulty breathing from allergies', 'sneezing fits due to pollen', 'allergic asthma', 'seasonal allergic reactions', 'itchy nose', 'nasal discharge from allergies',
        'blocked sinuses', 'itchy throat from allergies', 'dry throat from allergies', 'allergy flare-up', 'anaphylactic reaction', 'anaphylaxis', 'allergic dermatitis', 'rashes from allergens', 'swelling of lips',
        'swollen tongue', 'red eyes from allergies', 'tearing eyes from allergies', 'itchy and watery eyes'
    ],
    'fever': [
        'high temperature', 'elevated body temperature', 'feeling feverish', 'fevering', 'running a fever', 'burning up', 'feeling internally hot', 'having a temperature', 'spiking a fever', 'febrile state',
        'raised core temperature', 'overheated body', 'intense body heat', 'thermal imbalance', 'body overheating', 'raging fever', 'heated condition', 'abnormally warm body', 'pyrexia', 'uncontrolled internal heat',
        'feeling aflame', 'body heat surging', 'hot to the touch', 'internal ignition of warmth', 'body temperature surging', 'excessive warmth inside', 'bodily heat overload', 'intense flush', 'thermometer reading high',
        'scorching internal climate', 'burning sensation from within', 'sweltering body feel', 'thermal elevation', 'heated bloodstream', 'furnace-like feeling', 'feeling like an oven', 'heat radiating under skin',
        'internal fire', 'ignited from the inside', 'excessive internal warmth', 'body boiling over', 'incendiary sensation', 'intense internal glow', 'unrelenting heat', 'blazing warmth','feeling hot',
        'molten interior heat', 'near boiling point', 'incapacitating heat', 'relentless feverishness', 'sizzling body temp', 'flaming sensation', 'constant burning feeling', 'heat wave inside me', 'sweating due to internal heat',
        'red-hot core', 'smoldering embers of warmth', 'furnace-like core', 'pulsating heat', 'unremitting temperature rise', 'searing body condition', 'fire coursing through veins', 'endlessly hot', 'elevated reading on the thermometer',
        'no relief from heat', 'intense internal burning', 'volcanic warmth', 'torched from inside', 'superheated body', 'radical temperature spike', 'roasting sensation', 'tropical internal climate', 'heat-induced misery',
        'stoked internal fires', 'hothouse conditions inside', 'stifling fever fire', 'blazing internal inferno', 'relentless temperature climb', 'fever wave'
    ],
    'cough': [
        'Persistent cough', 'hacking cough', 'dry cough', 'wet cough', 'productive cough (with phlegm)', 'barking cough', 'non-productive cough', 'chronic cough',
        'coughing up mucus/sputum/blood', 'irritating cough', 'scratchy cough', 'whooping cough-like sound', 'continuous throat clearing', 'raspy hacking', 'chesty cough',
        'rattling cough', 'deep-chested cough', 'shallow annoying cough', 'tickling cough', 'lingering throat hack', 'spasm-like coughs', 'throaty expulsions',
        'worrisome coughing fits', 'repetitive cough bursts', 'phlegmy hacking', 'bronchial coughing', 'stubborn cough', 'dry tickling cough', 'persistent throat tickle',
        'strangling cough', 'wheezing cough', 'loud barking cough', 'cracking cough', 'sputum-laden cough', 'cough with gagging', 'spasmodic cough', 'stubborn dry cough',
        'overwhelming coughing sensation', 'sharp, dry cough', 'cough with sharp throat pain', 'violent coughing fits', 'painful coughing episodes', 'coughing after exertion',
        'chronic phlegm cough', 'intense wheezing cough', 'grating cough', 'wet chesty cough', 'gurgling cough'
    ],
    'sore throat': [
        'scratchy throat', 'painful throat', 'burning throat', 'irritated throat', 'swollen throat', 'inflamed throat', 'throat discomfort', 'throat scratchiness',
        'raw throat', 'tight throat', 'feeling of something stuck in throat', 'hoarse throat', 'swollen tonsils', 'throat inflammation', 'red throat', 'sore and swollen throat',
        'gritty throat', 'tender throat', 'raspy throat', 'dry throat', 'throat burning sensation', 'feeling of throat swelling', 'pain on swallowing', 'raw feeling in throat',
        'sore feeling when talking', 'throat soreness', 'painful swallowing', 'constant throat irritation', 'throat muscle soreness', 'tight feeling in throat',
        'throat dryness', 'itchy throat', 'burning sensation in throat', 'scratching feeling in throat', 'tenderness in throat', 'chronic throat discomfort', 'raspiness in voice',
        'feeling like throat is closing', 'constant need to clear throat', 'sore throat with hoarseness', 'dry cough with sore throat', 'sharp throat pain'
    ],
    'stomach pain': [
        'stomach pain', 'stomach ache', 'abdominal pain', 'belly ache', 'intestinal discomfort', 'stomach cramps', 'nauseous stomach pain',
        'sharp stomach pain', 'stomach tenderness', 'sharp abdominal cramps', 'stomach upset', 'abdominal tenderness', 'intestinal bloating', 'tummy pain', 'swollen belly',
        'feeling of fullness', 'feeling heavy in stomach', 'digestive pain', 'stomach spasms', 'soreness in abdomen', 'nausea and stomach ache',
        'gastric pain', 'pain after eating', 'belly discomfort', 'gurgling stomach', 'stomach churning', 'sharp abdominal pain', 'dull abdominal pain',
        'abdominal tightness', 'aching belly', 'painful digestion', 'pain under ribs', 'discomfort after meals',
        'uncomfortable stomach', 'intestinal cramps', 'sharp pain in lower abdomen', 'feeling of indigestion', 'pain around stomach area', 'belly pain', 'pain in the abdomen', 'stomach discomfort',
        'sharp stomach pain', 'dull abdominal pain', 'cramping in the abdomen', 'bloating with pain',
        'gas pain in the abdomen', 'stabbing pain in the belly', 'abdominal cramps', 'sharp pain in the stomach area', 'pain from indigestion', 'pain after eating', 'nauseating abdominal pain',
        'pain from gas buildup', 'pressure in the stomach', 'pain from constipation', 'distended abdomen', 'pain from ulcers', 'pain from bloating', 'pain from food intolerance',
        'sore stomach', 'pain from intestinal issues', 'gastrointestinal pain', 'tenderness in the stomach', 'pain near the navel', 'pain from diarrhea', 'stomach flu pain', 'pain in the lower abdomen',
        'feeling of fullness with pain', 'pain in the upper abdomen', 'stomach cramping', 'sharp abdominal cramps', 'nausea with stomach pain', 'abdominal swelling with pain', 'abdominal pain',
        'chronic stomach pain', 'pain with digestive issues', 'pain from food poisoning', 'pain from gallbladder issues', 'pain from acid reflux', 'pain in stomach', 'burning in stomach', 'itching in stomach', 'numbness in stomach',
        'stomach discomfort', 'ache in stomach', 'trouble in stomach', 'stomach cramp', 'sharp pain in stomach', 'pain in the stomach', 'stomach is paining', 'stomach is burning','stomach is aching'
    ],
    'weakness': [
        'tiredness', 'extreme tiredness','weariness', 'fatigued feeling', 'lack of energy', 'physical depletion', 'mental fatigue', 'chronic tiredness',
        'drained', 'feeling wiped out', 'feeling run down', 'fatigue','low energy', 'severe fatigue', 'feeling sluggish', 'morning fatigue', 'fatigue after exertion',
        'debilitating tiredness', 'drowsiness', 'chronic fatigue syndrome', 'feeling lethargic', 'mental sluggishness', 'physical tiredness', 'difficulty keeping eyes open',
        'lack of vitality', 'energy depletion', 'feeling drained', 'listlessness', 'burned out', 'exhausted state', 'feeling zoned out', 'tired all the time', 'fatigued state',
       'constant tiredness', 'no motivation', 'fatigued muscles', 'endless tiredness', 'lethargic movements',
        'lacking strength', 'body fatigue',  'feeling disconnected'
    ],
    'nausea': [
        'feeling nauseous', 'upset stomach', 'queasy', 'stomach turning', 'sick feeling', 'feeling like vomiting', 'gagging sensation', 'discomfort in stomach', 'unsettled stomach',
        'vomit-like sensation', 'stomach churn', 'sick to stomach', 'nauseous feeling', 'spinning stomach', 'intense nausea', 'gagging feeling', 'feeling on the verge of throwing up',
        'uneasy stomach', 'feeling faint', 'upset belly', 'dizzy stomach', 'nauseous and dizzy', 'headache and nausea', 'intense queasiness', 'morning sickness feeling',
        'stomach discomfort', 'feeling faint with nausea', 'stomach churn', 'constant nausea', 'puking feeling', 'nausea after eating', 'feeling like you could throw up',
        'stomach upset with nausea', 'unsettled feeling in stomach', 'feeling lightheaded with nausea', 'nausea with dizziness', 'craving nausea', 'nausea from food', 'stomach unease',
        'sick feeling after meals', 'swirling stomach', 'nauseous waves', 'gag reflex activated', 'gurgling stomach with nausea'
    ],
    'dizziness': [
        'Lightheadedness', 'feeling faint', 'woozy sensation', 'spinning feeling', 'off-balance', 'unsteady', 'dizzy spells', 'giddy feeling', 'vertiginous sensation',
        'wobbly feeling', 'swaying in mind', 'head swimming', 'feeling as if room is turning', 'disoriented equilibrium', 'teetering sense', 'tipsy sensation without alcohol',
        'floating head', 'unstable ground feeling', 'swirling environment', 'sense of being on a boat', 'nauseating spin', 'loss of spatial orientation', 'drifting balance',
        'feeling like I might topple', 'wavy floor sensation', 'heady unsteadiness', 'murky equilibrium', 'airy head sensation', 'constant near-tip-over feeling', 'mental wobble',
        'feathery balance', 'gravity shifting under feet', 'dizziness waves', 'rocking sensation', 'seasick feeling on land', 'fuzzy-headed instability', 'inner ear imbalance feeling',
        'wavy-field-of-view sensation', 'lurching environment', 'faltering steadiness', 'delicately balanced but slipping', 'rubbery legs feeling', 'giddy swirl in head',
        'tilting world', 'swaying sensation', 'imbalance feeling', 'shaky equilibrium', 'floating dizziness', 'spinning sensation', 'feeling off-kilter'
    ],

    'shortness of breath': [
        'Shortness of breath', 'breathlessness', 'difficulty breathing', 'feeling air hunger', 'fast breathing', 'shallow breathing', 'gasping for air',
        'labored breathing', 'struggling to breathe', 'tightness in chest while inhaling', 'feeling suffocated', 'cannot catch my breath', 'panting heavily', 'air feeling thin',
        'lungs working overtime', 'chest feels restricted', 'fighting for each breath', 'difficulty in breathing', 'strained respiration', 'feeling smothered', 'desperate for oxygen',
        'winded easily', 'constant puffing', 'breathing feels blocked', 'inhaling with effort', 'forced breathing', 'constant need to gulp air', 'sensation of drowning in open air',
        'chest heaviness on breathing', 'incomplete lung expansion', 'inadequate airflow', 'lungs not filling properly', 'needing to breathe harder', 'stuck in half-breath',
        'breath cut short', 'huffing and puffing', 'shallow panting', 'frantic search for air', 'hyperventilating feeling', 'feeling as if air is too thick', 'minimal air exchange',
        'muscle effort just to breathe', 'chest oppression', 'suffocating sensation even in open space', 'feeling strangled by lack of air', 'restrictive breathing pattern',
        'breathing feels like pushing through a straw', 'air-starved lungs', 'cannot take a deep breath', 'strained oxygen intake', 'feeling like each breath is a struggle',
        'never fully satisfied inhalation', 'gasping between words', 'needy breathing pattern', 'barely pulling in enough air', 'lungs working at half capacity', 'respiratory distress',
        'continuous short-windedness'
    ],
   'rapid breathing': [
         'heavy breathing', 'shallow breathing', 'heart skipping beats'
    ],
    'muscle pain': [
        'muscle ache', 'muscle soreness', 'muscle strain', 'muscle discomfort', 'muscle stiffness', 'muscle tension', 'muscle fatigue', 'muscle injury', 'muscle cramps',
        'muscle spasm', 'muscle pulling', 'muscle tears', 'muscle tightness', 'muscle throbbing', 'aching muscles', 'sore muscles', 'tender muscles', 'painful muscles',
        'muscle inflammation', 'deep muscle pain', 'sharp muscle pain', 'pulling sensation in muscles', 'muscle tenderness', 'delayed onset muscle soreness (DOMS)', 'straining muscle',
        'muscle weakness', 'fatigued muscles', 'muscle stiffness after exercise', 'muscle burning', 'swollen muscles', 'muscle discomfort on movement', 'muscle ache after exertion',
        'overused muscles', 'fatigue-related muscle pain', 'chronic muscle pain', 'localized muscle pain', 'muscle strain from overuse', 'aching from tension in muscles',
        'muscle soreness from heavy lifting', 'muscle discomfort from exercise', 'muscle pain after activity', 'muscle distress', 'inflamed muscle tissue', 'muscle spasm after effort',
        'tensed muscles', 'muscle overextension', 'pain in the calves', 'pain in the upper arms', 'pain in the back muscles'
    ],
    'insomnia': [
        'difficulty sleeping', 'trouble sleeping', 'sleeplessness', 'restlessness at night', 'inability to fall asleep', 'waking up during the night', 'frequent wake-ups',
        'early morning wakefulness', 'poor sleep quality', 'sleep deprivation', 'sleep disturbance', 'trouble staying asleep', 'sleep interruptions', 'unable to sleep through the night',
        'insufficient sleep', 'lack of sleep', 'unrefreshing sleep', 'tossing and turning', 'unsettled sleep', 'sleep issues', 'chronic insomnia', 'difficulty achieving deep sleep',
        'waking up too early', 'difficulty with sleep onset', 'difficulty getting comfortable at night', 'sleep anxiety', 'sleeping problems', 'frequent nighttime awakenings', 'irregular sleep cycle',
        'poor sleep habits', 'nighttime restlessness', 'waking in the middle of the night', 'sleep deprivation symptoms', 'daytime sleepiness from poor sleep', 'sleep fragmentation',
        'restless sleep', 'persistent insomnia', 'sleep troubles', 'light sleeping', 'short sleep duration', 'restorative sleep deprivation', 'fatigue from sleeplessness',
        'waking up exhausted', 'sleep cycle disruption', 'sleep onset difficulty', 'insomnia due to stress', 'mental hyperactivity preventing sleep', 'cannot sleep', 'unable to sleep','not able to sleep',
        'unable to fall asleep', 'not able to fall asleep'
    ],
    'rash': [
        'skin rash', 'redness on skin', 'skin irritation', 'skin inflammation', 'skin breakout', 'itchy rash', 'hives', 'blotchy skin', 'skin eruption', 'skin lesions',
        'red bumps on skin', 'inflamed skin', 'patchy rash', 'discolored skin', 'raised rash', 'painful rash', 'rash with blisters', 'dry rash', 'moist rash', 'allergic rash',
        'eczema', 'psoriasis patches', 'contact dermatitis', 'hives breakout', 'heat rash', 'prickly heat', 'scaly rash', 'rash on face', 'body rash', 'rashes on arms',
        'welts on skin', 'itchy patches on skin', 'skin redness', 'chronic skin rash', 'dry, scaly rash', 'blistering rash', 'swollen rash', 'rash with swelling',
        'inflamed, sore rash', 'rash with pus', 'pimple-like rash', 'rash caused by allergic reaction', 'skin irritation with swelling', 'flaky rash', 'raw skin from rash',
        'horrible itching rash', 'rashes from medication', 'painful itching on skin', 'burning sensation from rash'
    ],
    'congestion': [
        'nasal congestion', 'blocked nose', 'stuffy nose', 'clogged nasal passages', 'nasal obstruction', 'sinus congestion', 'sinus blockage', 'stuffy sinuses', 'pressure in sinuses',
        'nasal blockage', 'swollen nasal passages', 'congested sinuses', 'nose congestion', 'nasal stuffiness', 'head congestion',
        'swelling of nasal tissues', 'sinus pressure', 'stuffy feeling in head', 'congestion in sinus cavities', 'nasal stuffy feeling',
        'inflamed nasal passages', 'feeling of a blocked nose', 'swollen nostrils', 'nasal airway blockage', 'heavy feeling in head from congestion', 'sinus drainage blockage',
        'clogged airways', 'full nose', 'stuffy head', 'excess mucus in nose', 'thick mucus in nostrils', 'nasal obstruction from mucus', 'inability to breathe through nose',
        'nasal phlegm buildup', 'blocked airways', 'increased mucus production', 'congested nasal lining', 'swelling in nasal cavity', 'unpleasant nose feeling from congestion',
        'nasal fullness', 'pressure behind the eyes from congestion', 'nasal sinus blockage', 'nasal breathing difficulties'
    ],
    'runny nose': [
        'nasal discharge', 'drippy nose', 'clear runny nose', 'watery nose', 'excessive mucus secretion', 'nose dripping', 'watery nasal discharge', 'runny mucus from nose',
        'frequent nose blowing', 'excessive snot', 'thin nasal discharge', 'clear mucus', 'constant nose drip', 'streaming nose', 'watery runny nose', 'mucus dripping down from nose',
        'nose running uncontrollably', 'sticky nasal discharge', 'clear discharge from nostrils', 'frequent nasal wiping', 'constant nasal leaks', 'draining sinuses',
        'runny nose due to allergies', 'constant nasal secretions', 'wet nose', 'nose discharge', 'sinus leakage', 'flowing nose', 'uncontrolled nasal discharge',
        'persistent runny nose', 'dripping from nostrils', 'clogged but dripping nose', 'excessive mucus from nostrils', 'sniffling from a runny nose', 'constant nasal drip',
        'dripping sinuses', 'runny nose caused by cold', 'mucus continuously dripping', 'snotty nose', 'stuffy nose with runny discharge', 'chronic runny nose', 'dripping all day long'
    ],
    'sneezing': [
        'Sneezing fits', 'frequent sneezing', 'sneezing spells', 'sneezing bouts', 'sneezing attacks', 'sneezing episodes', 'uncontrollable sneezing', 'explosive sneezes',
        'repetitive sneezes', 'unstoppable nasal explosions','serial sneezing', 'sneeze after sneeze', 'chain-sneezing', 'nasal expulsions',
        'nasal reflex outbursts', 'convulsive sneezing', 'rapid-fire sneezes', 'machine-gun sneezing', 'persistent nasal expulsions', 'surprise sneezes',
        'tickling in nose triggering sneezes', 'staccato sneezing', 'sneeze cascades', 'recurrent sneezing', 'violent sneezing', 'spontaneous sneezes', 'sudden sneezing',
        'blasting sneezes', 'paroxysmal sneezing', 'intense sneezing', 'frequent sneezing attacks', 'sneezing with watery eyes', 'sudden fit of sneezing', 'uncontrollable nasal reflex',
        'hayfever sneezing', 'sneeze bursts', 'non-stop sneezing', 'gasping after sneezing', 'nasal reflex reactions', 'irritated sneezing', 'allergic sneezing', 'multiple sneeze cycles'
    ],
    'swollen lymph nodes': [
        'swollen glands', 'lymph node swelling', 'enlarged lymph nodes', 'swelling in neck', 'lumps in neck', 'tender lymph nodes', 'painful lymph nodes', 'swelling near jaw',
        'lymphatic swelling', 'lymph node enlargement', 'swollen glands under arms', 'underarm lymph node swelling', 'swollen neck glands', 'increased lymph node size',
        'lymphatic system swelling', 'lumps under the skin', 'swollen lymphatic glands', 'painful lumps in neck', 'inflamed lymph nodes', 'lymph node tenderness', 'neck swelling',
        'uncomfortable lumps in neck', 'tender neck lumps', 'swollen lymph glands in groin', 'swollen lymph nodes in armpit', 'painful swelling in neck', 'inflamed glands',
        'lymph node tenderness under jaw', 'enlarged glands in the throat', 'neck lumps', 'swollen lymph nodes behind ears', 'tender swollen glands', 'neck lymphatic swelling',
        'swelling in the throat', 'pain in swollen glands', 'pain from swollen lymph nodes', 'inflamed and tender lymph nodes', 'lymphatic swelling with pain'
    ],
    'joint pain': [
        'joint ache', 'joint discomfort', 'joint inflammation', 'joint stiffness', 'joint tenderness', 'pain in joints', 'arthritic pain', 'swollen joint', 'joint soreness',
        'joint irritation', 'musculoskeletal pain', 'painful joints', 'joint stiffness', 'grating joint feeling', 'aching joints', 'joint tightness', 'joint swelling', 'rheumatoid pain', 'stiff joints',
        'painful knees', 'painful shoulders', 'pain in elbows', 'wrist joint pain', 'ankle joint pain', 'hip joint pain', 'persistent joint pain', 'severe joint pain', 'uncomfortable joint pressure',
        'popping joints', 'clicking joints', 'cracking joints', 'sore knees', 'joint inflammation in fingers', 'inflamed joints', 'stiffened knee joints', 'swollen ankles', 'excessive joint pain',
        'joint tenderness', 'joint soreness from strain', 'arthralgia', 'aching knees', 'sharp joint pain', 'stabbing joint pain', 'chronic joint ache', 'inflamed elbow joints',
        'chronic knee pain', 'joint damage', 'strained joint', 'degenerative joint disease', 'discomfort in joints', 'dull joint ache', 'acute joint pain', 'swollen hands', 'weakening joint flexibility',
        'muscle and joint discomfort', 'continuous joint pain', 'painful back joints', 'arthritic inflammation', 'joint locking', 'joint clicking', 'joint popping', 'joint dislocation'

    ],
   'diarrhea': [
        'loose stools', 'frequent bowel movements', 'watery stools', 'runny stools', 'loose bowels', 'urgent need to defecate', 'watery bowel movements', 'explosive diarrhea',
        'stomach upset with diarrhea', 'frequent trips to the bathroom', 'diarrhea with cramping', 'abnormal stool consistency', 'watery feces', 'fecal urgency', 'loose bowel movement',
        'urgent diarrhea', 'persistent diarrhea', 'morning diarrhea', 'stomach flu diarrhea', 'digestive distress', 'frequent liquid stools', 'runny bowel movements', 'intense bowel movements',
        'diarrheal episode', 'loose stool rush', 'urgent diarrhea attack', 'acute diarrhea', 'chronic diarrhea', 'pale watery stools', 'stomach churn with diarrhea', 'intestinal upset',
        'frequent bowel clearing', 'fluid-filled stools', 'non-stop diarrhea', 'gut infection diarrhea', 'dehydrating diarrhea', 'uncontrolled liquid stools', 'loose stool frequency',
        'constantly running to the bathroom', 'liquid-filled intestines', 'intense gastrointestinal upset', 'abnormally frequent bowel movements', 'severe bowel looseness', 'bowel irregularity',
        'involuntary liquid stools', 'gassy diarrhea', 'splashy diarrhea', 'digestive upset causing liquid stools', 'diarrhea with abdominal pain'
    ],
    'vomiting': [
        'throwing up', 'puking', 'stomach upset', 'retching', 'emesis', 'nausea with vomiting', 'forcefully throwing up', 'heaving', 'vomiting episodes', 'sick stomach',
        'gagging', 'expelling stomach contents', 'stomach expulsion', 'violent vomiting', 'repeated vomiting', 'uncontrollable vomiting', 'upchucking', 'spitting up', 'retching reflex',
        'forceful expulsion of food', 'involuntary stomach release', 'emetic response', 'feeling of needing to vomit', 'gag reflex triggering', 'chronic vomiting', 'severe vomiting',
        'nausea-induced vomiting', 'unpleasant stomach eruption', 'stomach contents expelled forcefully', 'gastrointestinal purge', 'expulsion of gastric contents', 'violent heaving',
        'nauseated vomiting', 'vomit-induced gagging', 'stomach-purging sensation', 'retching uncontrollably', 'throwing up after eating', 'puking episodes', 'sick and throwing up',
        'puking from irritation', 'regurgitating food', 'empty stomach vomiting', 'morning sickness vomiting', 'nausea attacks with vomiting', 'emesis due to motion sickness', 'heaving up'
    ],

    'ear pain': [
        'ear ache', 'pain in the ear', 'ear discomfort', 'ear irritation', 'painful ear', 'throbbing ear ache', 'sharp ear pain', 'dull ear pain', 'stabbing pain in ear', 'ringing ear pain',
        'pressure in ear', 'ear sensitivity', 'intense ear discomfort', 'itchy ear', 'swollen ear', 'ear tenderness', 'ear pulsations', 'persistent ear pain', 'ear infection pain',
        'ear tenderness', 'pain behind ear', 'soreness in ear', 'ear pressure', 'ear inflammation', 'ear ache from cold', 'stuffy ear pain', 'pain in ear canal', 'ear ache when swallowing',
        'painful inner ear', 'hearing sensitivity with pain', 'fluid in ear causing pain', 'acute ear pain', 'chronic ear ache', 'pain after water exposure', 'ear infection causing pain',
        'tender ear lobes', 'painful eardrum', 'painful earful feeling', 'pounding ear pain', 'sharp stabbing ear ache', 'swollen ear canal', 'eardrum sensitivity', 'sharp pressure sensation in ear',
        'soreness in ear cavity', 'clogged ear with pain', 'throbbing sensation in ear', 'ear ache during sleep'
    ],
    'back pain': [
        'lower back pain', 'upper back pain', 'spinal pain', 'pain in the back', 'back is paining', 'achy back', 'sharp back pain', 'dull back pain', 'severe back pain', 'chronic back pain',
        'stiff back', 'muscle soreness in back', 'pressure in lower back', 'pain between shoulder blades', 'sharp pain in spine', 'pain in back muscles', 'backache from lifting',
        'back discomfort', 'spinal discomfort', 'pain in lumbar region', 'back injury', 'radiating back pain', 'tight back muscles', 'spinal stiffness', 'lower back strain', 'back pain after exercise',
        'muscle strain in the back', 'burning sensation in back', 'intense back pain', 'nagging back pain', 'sharp stabbing pain in lower back', 'back tension', 'sore spine',
        'pinched nerve in back', 'back spasms', 'pain when bending', 'pain while standing up', 'pressure in upper back', 'burning pain in the back', 'pain in the sacral region',
        'pain with movement', 'back pain when sitting', 'lower back discomfort', 'muscular back pain', 'upper spinal discomfort', 'radiating pain down the back'
    ],
    'cold': [
        'Common cold', 'head cold', 'mild viral infection', 'slight sniffles', 'catching a cold', 'seasonal cold', 'chest cold', 'light upper respiratory infection', 'mild sniffle bug',
        'standard cold virus', 'low-grade nasal virus', 'mild runny-nose ailment', 'basic rhinovirus', 'everyday cold symptoms', 'short-term sniffles', 'routine winter bug', 'easy viral cold',
        'minor head stuffiness illness', 'typical seasonal illness', 'cold symptoms', 'stuffy nose cold', 'mild sore throat with cold', 'cold with slight fever', 'cough with cold',
        'runny nose cold', 'sneezing with cold', 'mild chest congestion', 'low-grade cold infection', 'itchy throat cold', 'general cold symptoms', 'nasal congestion from cold',
        'watery eyes with cold', 'mild head congestion', 'cold-related fatigue', 'chilly viral infection', 'upper respiratory cold', 'typical cold symptoms', 'stuffy feeling from cold',
        'cough and cold', 'runny nose from cold', 'frequent sneezing cold', 'cold-related chills', 'feeling chilled from cold', 'aching muscles with cold', 'minor fever with cold',
        'slight cold discomfort', 'cold-induced sore throat' ,'feeling cold'
    ],
    'sweat': [
        'sweating', 'excessive sweating', 'unusual sweating', 'profuse sweating', 'drenched in sweat', 'perspiring heavily', 'sweating buckets', 'clammy sweating', 'dripping perspiration',
        'bead-like sweat on skin', 'moisture streaming down face', 'uncontrollable sweating', 'soaked in sweat', 'overactive sweat glands', 'sweaty and damp skin', 'sweat-soaked clothes',
        'constant perspiration', 'sticky sweat', 'salty perspiration', 'glistening with sweat', 'sweat trickling down spine', 'nervous sweating', 'stress-induced sweat', 'drenching perspiration',
        'sweat-laden body', 'humid feeling', 'slick skin', 'warm moisture on skin', 'sweat beads forming everywhere', 'bodily moisture overload', 'persistent dampness', 'sweaty palms and forehead',
        'rivers of sweat', 'sweat dripping off hairline', 'sweat-soaked sheets', 'nocturnal sweating', 'smelly perspiration', 'standing in a pool of sweat', 'sweat forming under arms', 'shiny perspiring face',
        'sweat running down temples', 'sweat-induced chafing', 'slick and slippery feeling', 'sweating like in a steam room', 'permanent dampness', 'sweat stains on clothing'
    ],
    'swelling': [
        'swollen area', 'edema', 'swelling of body part', 'fluid retention', 'swollen body part', 'inflamed tissue', 'swollen limbs', 'puffiness',
        'swollen joints', 'swollen ankle', 'swollen hands', 'swollen feet', 'localized swelling', 'swollen skin', 'swelling in legs', 'swelling due to injury', 'swollen belly',
        'swollen face', 'swollen knees', 'edematous swelling', 'painful swelling', 'swollen extremities', 'swelling from infection', 'swelling from trauma', 'swelling after surgery',
        'swelling of the face', 'swelling under the skin', 'swollen throat', 'swelling with discomfort', 'puffy hands', 'swelling after a fall', 'generalized swelling', 'swelling in eyes',
        'swelling from arthritis', 'swelling around wounds', 'enlarged tissue area', 'swelling from allergic reaction', 'swelling in body cavity', 'swelling around the joints','bruising'
    ],
    'tremor': [
        'shaking', 'shivering', 'twitching', 'involuntary movements', 'nervous shaking', 'muscle tremors', 'rhythmic shaking', 'trembling hands', 'uncontrolled muscle movement',
        'shaking limbs', 'twitchy fingers', 'uncontrolled tremor', 'flickering motion', 'trembling body', 'shaky movements', 'muscle spasms', 'jerking', 'shivering body', 'shaky hands',
        'shaking from cold', 'nervous tremors', 'trembling sensation', 'shuddering', 'uncontrollable shaking', 'flickering muscles', 'twitching eyes', 'nervous jerks', 'shaky fingers',
        'twitching limbs', 'muscle jerks', 'nervous body shakes', 'subtle muscle tremors', 'involuntary shaking', 'feeling of tremors', 'trembling body parts', 'sporadic body shaking',
        'hand shaking', 'shaky voice', 'rhythmic tremors', 'shivering fingers', 'body quivering', 'body shudders', 'shaking from anxiety'
    ],
    'chills': [
        'Shivering', 'trembling with cold', 'goosebumps', 'feeling cold inside', 'uncontrollable shaking', 'teeth chattering', 'feeling frosty', 'quivering limbs', 'body shaking from cold',
        'icy tremors', 'frigid vibrations', 'quaking with chill', 'hair standing on end', 'trembling internally', 'spasmodic shivers', 'cold-induced tremble', 'chilled to the bone',
        'freezing sensation', 'vibrating with cold', 'small uncontrollable shakes', 'persistent shuddering', 'subtle shivers', 'prickly gooseflesh', 'frost-like feeling', 'quivery muscles',
        'rattled by chill', 'shudders running down spine', 'uncontrollable cold tremors', 'shaky fingers and toes', 'rattling teeth', 'jittering from cold', 'frigid trembles',
        'cold-induced shaking', 'body frozen in chills', 'deep chills', 'numbing cold'
    ],

    'eye pain': [
        'ocular pain', 'eye discomfort', 'pain in the eye', 'eye ache', 'sore eye', 'sharp pain in the eye', 'pain around the eyes', 'painful vision', 'pain behind the eye',
        'irritation in the eye', 'burning sensation in the eye', 'dry eye pain', 'stabbing eye pain', 'eye strain', 'pressure in the eye', 'throbbing in the eye',
        'sensitive eyes', 'eye tenderness', 'aching in the eye', 'eye inflammation', 'pulsing pain in the eye', 'intense eye discomfort', 'distorted vision from pain', 'foreign body sensation in the eye',
        'sharp eye ache', 'vision-related pain', 'severe eye pain', 'sharp stabbing pain in the eye', 'pain in the eyeball', 'tired eye pain', 'swollen eye discomfort', 'throbbing behind the eyes',
        'pain from light sensitivity', 'pain after reading', 'pain when blinking', 'gritty feeling in the eyes', 'intense eye pressure', 'pain around the eyelids', 'blurry vision with pain', 'puffy eyes with pain',
        'pain near the cornea', 'stinging pain in the eye', 'pain with redness in the eye', 'ocular discomfort', 'persistent eye pain', 'painful feeling when moving eyes', 'pressure sensation in the eyes',
        'pain from eye strain', 'pain with dry eyes', 'eye irritation', 'eye swelling', 'eye tearing','pain in eye','pain in the eyes', 'itching in eye', 'itching in eyes', 'burning in eye', 'burning in eyes',
        'sore eye', 'sore eyes', 'numbness in eye', 'numbness in eyes', 'eye discomfort', 'ache in eye', 'ache in eyes',
        'trouble in eye', 'trouble in eyes','pain in my eye','pain in eyes', 'pain in my eyes'
    ],
    'ear pain': [
        'ear ache', 'pain in the ear', 'ear discomfort', 'sharp ear pain', 'throbbing ear ache', 'ear irritation', 'pressure in the ear', 'sharp pain in the ear', 'stabbing ear pain',
        'pain in the eardrum', 'intense ear ache', 'pain around the ear', 'aching ear', 'pain in the ear canal', 'tinnitus-related ear pain', 'ear pressure', 'burning pain in the ear',
        'ear tenderness', 'feeling of fullness in the ear', 'swollen ear', 'pain behind the ear', 'ear congestion', 'pain with hearing loss', 'fluid-filled ear pain', 'pain from ear infection',
        'dull ear pain', 'ear pain with headache', 'pain from wax buildup', 'ear ache with cold', 'sharp sensation in the ear', 'sore ear', 'ringing ear pain', 'pain in the outer ear',
        'pain from sinus pressure', 'pain in inner ear', 'pain from ear trauma', 'pain from water exposure', 'sensitive ear', 'painful ear canal', 'deep ear pain', 'pain in the ear after flying',
        'uncomfortable ear ache', 'hearing discomfort', 'ear pain after loud noise', 'sensitive eardrum'
    ],
    'nose pain': [
        'pain in the nose', 'nasal pain', 'sinus pain', 'stabbing pain in the nose', 'sharp nasal discomfort', 'painful sinus area', 'throbbing pain in the nose', 'blocked nose pain',
        'pain from sinusitis', 'nasal congestion pain', 'pain from cold in the nose', 'swollen nose', 'pressure in the sinuses', 'pain from a cold', 'tenderness in the nose', 'painful nostrils',
        'pain from nasal polyps', 'pain around the nostrils', 'nosebleed-related pain', 'pain with nasal drip', 'stuffy nose pain', 'pain due to allergies', 'burning sensation in the nose',
        'painful nasal congestion', 'aching nose', 'chronic nasal pain', 'pain when breathing through nose', 'nose pressure', 'pain at the bridge of the nose', 'pain in nasal cavity', 'itchy nose with pain',
        'pain from external nose injury', 'pain from sniffles', 'swelling in the nasal area', 'facial pain near the nose', 'pain from sinus congestion', 'painful nasal passages', 'pain in the septum',
        'pain from sniffing'
    ],
    'throat pain': [
        'sore throat', 'pain in the throat', 'scratchy throat', 'throat discomfort', 'painful swallowing', 'irritated throat', 'dry throat pain', 'burning sensation in the throat', 'throat tenderness',
        'throat scratchiness', 'swollen throat', 'pain from tonsillitis', 'strep throat pain', 'pain from a cold', 'inflammation of the throat', 'sore throat with fever', 'pain in the tonsils',
        'pain from acid reflux', 'hoarse throat', 'tight throat pain', 'painful voice box', 'pain in the larynx', 'pain with coughing', 'pain with dry mouth', 'pain with difficulty swallowing',
        'pain in the pharynx', 'pain with swollen glands', 'throat congestion', 'pain after excessive talking', 'pain from dry air', 'pain from smoking', 'throat dryness', 'severe throat discomfort',
        'pain after eating', 'pain from post-nasal drip', 'sore and swollen throat', 'pain from throat infection', 'swollen tonsils with pain', 'tightness in the throat', 'pain from sore mouth',
        'stabbing throat pain', 'pain when swallowing food', 'burning throat pain', 'pain in throat', 'burning in throat', 'itching in throat', 'sore throat', 'numbness in throat',
        'throat discomfort', 'ache in throat', 'trouble in throat', 'throat irritation','throat is paining'
    ],
    'jaw pain': [
        'pain in the jaw', 'jaw discomfort', 'jaw ache', 'pain in the temporomandibular joint', 'TMJ pain', 'painful jaw muscles', 'pain from jaw clenching', 'tooth-related jaw pain',
        'sharp jaw pain', 'throbbing jaw pain', 'jaw tension', 'muscle pain in the jaw', 'pain from grinding teeth', 'jaw stiffness', 'pain in the lower jaw', 'pain from jaw injury', 'pain near the jawline',
        'pain around the mouth area', 'pain when chewing', 'discomfort in the jaw', 'jaw lock', 'jaw popping pain', 'pain around the ear and jaw', 'pain from dental issues', 'pain with jaw movement',
        'swollen jaw area', 'pain in the temporomandibular joint', 'facial pain near the jaw', 'pain from misaligned teeth', 'jaw pain from stress', 'jaw swelling', 'pain when yawning',
        'sharp pain in jaw joint', 'stiffness in jaw', 'dull aching jaw pain', 'pain from jaw trauma', 'pain from wisdom teeth', 'jaw clicking', 'pain in the side of the jaw', 'pain from jaw misalignment',
        'pain after jaw surgery', 'pain during biting', 'jaw discomfort while sleeping'
    ],
    'tooth pain': [
        'toothache', 'dental pain', 'pain in the tooth', 'sharp tooth pain', 'throbbing tooth pain', 'pain from cavity', 'pain from tooth infection', 'pain in the gums', 'sensitive teeth pain',
        'pain from a dental abscess', 'pain when chewing', 'pain from tooth decay', 'pain with tooth sensitivity', 'pain after dental work', 'pain in the tooth root', 'pain from tooth fracture',
        'pain from gum disease', 'tooth pressure', 'pain after eating', 'pain when brushing teeth', 'pain from wisdom teeth', 'pain in the molars', 'pain from misaligned teeth', 'dull tooth pain',
        'pain from tooth eruption', 'pain from a cracked tooth', 'pain with swollen gums', 'constant toothache', 'sharp shooting tooth pain', 'pain from tooth trauma', 'dental discomfort',
        'pain in the tooth nerve', 'pain from filling', 'gum-related tooth pain', 'pain from chipped tooth', 'pain from teeth grinding', 'pain from dental infection', 'pain from plaque buildup',
        'tooth pressure with pain', 'pain from oral sores', 'pain in upper teeth', 'severe tooth pain', 'pain in front teeth'
    ],
    'chest pain': [
        'pain in the chest', 'chest discomfort', 'tightness in chest', 'pressure in the chest', 'sharp chest pain', 'tight chest feeling', 'stabbing chest pain', 'burning chest pain', 'aching chest',
        'chest heaviness', 'pain in the breastbone', 'pain radiating from chest', 'dull chest pain', 'pain from heartburn', 'pain from acid reflux', 'pain in the ribs', 'pain in the upper chest',
        'sharp stabbing pain in chest', 'chest tightness', 'pain under the sternum', 'pain when breathing deeply', 'feeling of chest pressure', 'pain from pulmonary issues', 'heart-related chest pain',
        'sharp pain in the breastbone', 'radiating chest discomfort', 'pain when moving', 'pain from costochondritis', 'pain from muscle strain in chest', 'deep chest discomfort', 'pain from anxiety',
        'dull aching chest pain', 'pain in the upper left chest', 'pain when lying down', 'sore chest', 'pain from trauma to chest', 'persistent chest pain', 'discomfort after exercise',
        'pain in the center of the chest', 'pain from chest cold', 'pain in the chest while breathing', 'sore chest area', 'pain in the left side of the chest', 'pain from coughing', 'pain from deep breathing'
    ],

    'knee pain': [
        'knee discomfort', 'pain in the knee', 'joint pain in the knee', 'knee ache', 'sharp knee pain', 'throbbing knee pain', 'stabbing pain in the knee', 'pain in the knee joint',
        'pain from knee injury', 'pain from knee strain', 'knee swelling', 'pain when bending knee', 'pain while walking', 'pain after exercise', 'pain from knee overuse', 'pain with knee movement',
        'pain in the kneecap', 'pain on the outer knee', 'pain on the inner knee', 'pain from ligament injury', 'pain from torn meniscus', 'sharp pain in the knee cap',
        'knee joint inflammation', 'pain when climbing stairs', 'pain with swelling', 'pain from running', 'pain from twisting knee', 'pain when standing up', 'knee tenderness', 'pain from patella dislocation',
        'pain with knee instability', 'pain from bursitis', 'pain with osteoarthritis', 'pain from cartilage damage', 'pain after knee surgery', 'pain in the back of the knee', 'numbness in knee', 'numbness in knees', 'itching in knee', 'itching in knees',
        'knee discomfort', 'ache in knees', 'sore knee', 'sore knees', 'knee burning', 'burning sensation in knee',
        'tingling in knees', 'trouble in knee', 'trouble in knees'
    ],
    'foot pain': [
        'pain in the foot', 'plantar pain', 'foot discomfort', 'foot ache', 'pain in the heel', 'sharp foot pain', 'throbbing foot pain', 'pain from foot injury', 'pain in the arch',
        'pain from flat feet', 'pain from bunions', 'pain in the toes', 'pain from corns', 'pain from calluses', 'pain from foot fractures', 'pain from wearing tight shoes', 'pain with walking',
        'pain from arthritis in foot', 'swollen foot', 'pain in the sole', 'pain when standing', 'pain from sprained ankle', 'pain from tendinitis', 'sharp pain in the foot arch', 'pain in foot joints',
        'pain in the ball of the foot', 'numbness in foot with pain', 'heel pain', 'pain from Morton’s neuroma', 'pain in foot after exercise', 'pain from overuse', 'pain after running', 'foot cramping pain',
        'pain after standing for long periods', 'pain in the toes after walking', 'sharp heel pain', 'foot pain from nerve issues', 'pain from diabetic neuropathy', 'foot pain from swelling', 'pain after wearing heels'
    ],
    'ankle pain': [
        'ankle discomfort', 'pain in the ankle', 'twisted ankle pain', 'pain from sprained ankle', 'swollen ankle', 'sharp ankle pain', 'throbbing pain in the ankle', 'pain when walking',
        'pain after ankle injury', 'pain from overuse', 'pain after exercise', 'pain with ankle movement', 'pain with swelling', 'pain from torn ligament', 'pain in the outer ankle', 'pain in the inner ankle',
        'pain in the ankle joint', 'pain from ligament strain', 'pain from ankle fracture', 'ankle tenderness', 'pain with ankle instability', 'pain when standing', 'sharp pain in the ankle',
        'pain in ankle tendon', 'pain after running', 'pain from ankle arthritis', 'pain with twisting', 'pain in ankle after jumping', 'pain in the Achilles tendon', 'stabbing pain in ankle',
        'pain with ankle sprain', 'ankle bruising', 'pain when walking on uneven surfaces', 'pain when bending the foot', 'pain in the heel of the ankle', 'pain during sports activities', 'pain when stretching ankle'
    ],
     'wrist pain': [
        'pain in the wrist', 'wrist discomfort', 'carpal pain', 'wrist ache', 'pain in the wrist joint', 'wrist inflammation', 'swollen wrist', 'stiff wrist pain', 'pain from repetitive strain',
        'tenderness in the wrist', 'sharp wrist pain', 'throbbing wrist pain', 'pain after wrist injury', 'pain from wrist overuse', 'wrist sprain pain', 'pain in the wrist tendons', 'wrist joint stiffness',
        'pain with wrist movement', 'pain during wrist flexion', 'pain in wrist ligaments', 'carpal tunnel syndrome pain', 'pain from arthritis in the wrist', 'pain with wrist rotation', 'pain in the wrist after typing',
        'wrist discomfort from injury', 'pain around wrist bones', 'dull wrist pain', 'pain when lifting objects', 'pain in the wrist after exertion', 'pain from wrist fractures', 'wrist tendonitis pain',
        'pain after hand movements', 'pain in the wrist when gripping', 'pain with wrist extension', 'stiff wrist from overuse', 'sharp sensation in the wrist', 'pain after extended typing', 'pain with wrist bending',
        'swollen joints in the wrist', 'wrist pain with numbness', 'pain around wrist bones after activity', 'pain in the carpal region', 'wrist discomfort with tingling sensation', 'pain after sports activity'
    ],
    'hand pain': [
        'hand discomfort', 'pain in the hand', 'aching hand', 'sharp hand pain', 'throbbing hand pain', 'pain in hand joints', 'pain in the palm of the hand', 'pain in the fingers', 'pain in the knuckles',
        'pain from hand injury', 'swollen hand', 'pain with hand movement', 'hand strain pain', 'numbness in the hand', 'pain after gripping', 'pain in the thumb', 'pain from arthritis in the hand',
        'pain from repetitive motions', 'pain from carpal tunnel syndrome', 'pain from hand overuse', 'hand joint pain', 'pain from hand sprain', 'pain from tendonitis in the hand', 'pain in the wrist with hand use',
        'sharp pain when holding objects', 'burning pain in the hand', 'painful hand cramps', 'dull hand pain', 'pain from hand fracture', 'pain from swelling in the hand', 'joint stiffness in the hand',
        'pain from typing', 'pain after using the hand excessively', 'pain when stretching the hand', 'pain in hand from trauma', 'pain when writing', 'pain with hand dexterity', 'pain in the hand after exercise',
        'muscle pain in the hand', 'pain from cold in the hand', 'pain after lifting objects', 'pain in the palm', 'pain in the wrist', 'hand soreness', 'pain from hand injury',
        'pain from overuse', 'pain from arthritis', 'pain while gripping', 'hand muscle pain', 'pain with swelling', 'pain in the hands', 'numbness in hand', 'numbness in hands', 'itching in hand', 'itching in hands',
        'burning in hand', 'burning in hands', 'sore hand', 'sore hands', 'hand discomfort', 'ache in hand', 'ache in hands',
        'trouble in hand', 'trouble in hands','hands are aching','hand is aching', 'hand is paining', 'hands are paining'
    ],
    'arm pain': [
        'pain in the arm', 'upper limb pain', 'arm discomfort', 'sharp arm pain', 'throbbing arm pain', 'pain in the elbow', 'pain in the shoulder', 'pain in the forearm', 'pain in the biceps',
        'pain from arm injury', 'pain from repetitive arm movement', 'pain from tendonitis in the arm', 'muscle pain in the arm', 'nerve pain in the arm', 'pain from elbow strain', 'pain in the upper arm muscles',
        'pain from arm sprain', 'pain in the wrist and arm', 'stiffness in the arm', 'swollen arm', 'pain when moving the arm', 'burning pain in the arm', 'aching in the arm', 'arm cramping',
        'pain from lifting with the arm', 'pain when raising the arm', 'pain from overuse of the arm', 'pain from arm fracture', 'pain in the arm muscles after exercise', 'pain from muscle strain in the arm',
        'pain from joint inflammation', 'sharp pain in the arm muscles', 'pain in the elbow joint', 'pain in the shoulder joint', 'dull arm pain', 'pain in the forearm when lifting', 'shooting arm pain',
        'nerve-like pain in the arm','pain in the arm', 'pain in the arms'
    ],
    'leg pain': [
        'pain in the leg', 'lower limb pain', 'leg discomfort', 'muscle pain in the leg', 'pain in the thigh', 'pain in the calf', 'pain in the shin', 'pain from leg injury',
        'sharp leg pain', 'throbbing leg pain', 'aching leg pain', 'pain in the leg muscles', 'pain in the leg joints', 'pain when walking', 'pain from leg cramps', 'pain after leg exercise',
        'pain after running', 'pain from overuse', 'pain in the hamstring', 'pain from leg sprain', 'muscle soreness in the leg', 'pain in the calf after activity', 'pain from leg fractures',
        'burning pain in the leg', 'pain from restless legs', 'pain when standing', 'pain in the thigh after sitting', 'pain in the foot and leg', 'pain with leg movement', 'pain from sciatica',
        'leg pain from sitting too long', 'pain when bending the leg', 'pain in the shin muscles', 'swollen leg', 'pain from arthritis in the leg', 'dull pain in the leg', 'sharp pain in the lower leg',
        'pain when walking on uneven ground', 'pain in the lower back and leg', 'pain in the legs', 'pain in leg', 'pain in legs', 'numbness in leg', 'numbness in legs', 'itching in leg', 'itching in legs',
        'burning in leg', 'burning in legs', 'sore leg', 'sore legs', 'leg discomfort', 'ache in leg', 'ache in legs',
        'trouble in leg', 'trouble in legs', 'leg cramp', 'sharp pain in leg'
    ],

    'mouth pain': [
        'pain in mouth', 'numbness in mouth', 'itching in mouth', 'burning in mouth', 'soreness in mouth',
        'discomfort in mouth', 'mouth ache', 'trouble in mouth'
    ],

    'confusion': [
        'disorientation', 'muddled thinking', 'mental fog', 'trouble thinking clearly', 'brain fog', 'cognitive cloudiness', 'puzzled state', 'jumbled thoughts', 'incoherent reasoning', 'tangled mental process',
        'unclear comprehension', 'befuddled mind', 'scrambled logic', 'perplexed state', 'hazy understanding', 'blurred mental picture', 'fuzzy reasoning', 'perplexity', 'baffled intellect',
        'uncertain grasp', 'foggy mental landscape', 'clouded judgment', 'unclear headspace', 'mixed-up thoughts', 'lack of mental clarity', 'distorted perspective', 'murky understanding',
        'minds in knots', 'head scrambled eggs feeling', 'no clear thread of thought', 'haphazard reasoning', 'bewildered stance', 'lost mental bearings', 'mental haze', 'unclear mental signals',
        'vague cognitive process', 'mental static', 'mentally adrift', 'diluted focus', 'no sharpness in mind', 'blinking confusion', 'unsure mental footing', 'perplexed awareness',
        'reduced mental acuity', 'messy mental white noise'
    ],
    'hip pain': [
        'pain in the hip', 'hip discomfort', 'hip joint pain', 'pain from hip arthritis', 'sharp hip pain', 'throbbing hip pain', 'pain in the hip joint', 'pain in the hip area', 'pain from hip injury',
        'hip inflammation', 'pain from hip strain', 'pain from bursitis in the hip', 'pain when moving the hip', 'pain from overuse of the hip', 'pain in the groin area', 'pain during walking',
        'pain from hip fracture', 'pain after standing for a long time', 'pain from hip surgery', 'pain when lying on the hip', 'pain with hip rotation', 'pain in the side of the hip',
        'pain in the front of the hip', 'pain from sciatica', 'sharp hip joint pain', 'pain from hip flexor strain', 'deep hip pain', 'pain from hip dislocation', 'aching hip pain',
        'pain from muscle strain around the hip'
    ],
    'back pain': [
        'lower back pain', 'upper back pain', 'spinal pain', 'pain in the back', 'back discomfort', 'sharp back pain', 'muscle pain in the back', 'pain in the lumbar region', 'pain in the thoracic region',
        'pain in the back muscles', 'dull back pain', 'throbbing back pain', 'pain when bending over', 'pain when standing', 'pain from sciatica', 'pain from a herniated disc', 'pain from poor posture',
        'pain from back injury', 'sharp pain in the lower back', 'pain from spinal stenosis', 'pain in the upper back', 'pain in the back after lifting', 'pain from overuse of back muscles',
        'chronic back pain', 'pain from spinal degeneration', 'aching back muscles', 'back pain with tingling', 'pain from back strain', 'pain when sitting for too long', 'swollen back muscles',
        'pain from muscle spasms in the back', 'pain with back movement', 'sharp shooting pain in the back', 'pain in back', 'numbness in back', 'itching in back', 'burning in back', 'sore back', 'back discomfort',
        'ache in back', 'trouble in back', 'back stiffness', 'sharp pain in back', 'back tenderness','back is paining', 'back pains', 'pain in back', 'pain in the back'
    ],
    'memory loss': [
        'forgetfulness', 'difficulty recalling', 'poor memory', 'memory lapses', 'amnestic episodes', 'short-term memory issues', 'difficulty remembering recent events', 'blanking out on details',
        'slip of the mind', 'fuzzy recollections', 'failing memory', 'losing track of thoughts', 'vacant mental storage', 'holes in memory', 'patchy recollection',
        'vanishing details from mind', 'gaps in remembrance', 'fleeting mental notes', 'mental blanks', 'fragmented memory', 'elusive past events', 'stuttering memory', 'hazy recall',
        'details fading away', 'mental erasures', 'unstable memory bank', 'shaky recollections', 'selective forgetfulness', 'mental blackouts', 'fuzzy mental snapshots', 'misplacing thoughts',
        'memory glitches', 'jumbled recall', 'inability to summon certain facts', 'feeling brain-drained', 'memory going dark', 'fragments of information missing', 'scattering of remembered info',
        'ghost-like recollections', 'losing the thread of events', 'names escaping me', 'scrambled memory patterns', 'drifting mental records', 'unreliable mental archives', 'evaporation of recent info',
        'dimming recollection', 'disintegrating memory', 'thinning retention', 'leaky mental container', 'short-circuited memory', 'mental fade-outs', 'mental sputtering', 'forgetting simple things',
        'intangible memories slipping away', 'memory weakening over time', 'groping for details', 'elusive truths once known', 'mental book pages going blank', 'unstable mental files',
        'dulled memory edges', 'uncertain memory foothold', 'eroded recollections', 'falling out of my mind', 'scattering mental fragments', 'temporary amnesia-like moments', 'no access to recent thoughts',
        'memory wires disconnected', 'stuttering recollection attempts', 'defragmented mental records', 'shaky mental camera', 'fading mental impressions', 'mind like a sieve', 'losing info instantly',
        'rattled mental library', 'concept slip-through', 'flickering data in mind', 'barren mental shelves', 'no retrieval of recent facts', 'thinking it’s on the tip of my tongue but never surfacing',
        'losing track of recent conversations', 'difficulty holding new info', 'memory short-circuits frequently', 'mental vacancy', 'ephemeral recollections', 'passing mental clouds with no retention',
        'drifting away from details', 'no anchor to past events'
    ],
  'hallucination': [
    'delusion', 'illusion', 'false perception', 'sensory distortion', 'visual disturbance', 'auditory hallucination',
    'perceptual distortion', 'false sensory experience', 'phantom perception', 'psychotic episode', 'imagined sight', 'imagined sound', 'mind illusion',
    'sensory misperception', 'hallucinatory experience', 'out-of-body experience', 'visual illusion', 'auditory illusion', 'mental delusion', 'altered reality'
],

  'loss of appetite': [
    'decreased appetite', 'reduced appetite', 'appetite loss', 'lack of appetite', 'poor appetite', 'no desire to eat', 'loss of interest in food', 'unwillingness to eat',
    'inability to eat', 'diminished appetite', 'eating less', 'loss of hunger', 'food aversion', 'food intolerance', 'decreased desire to eat', 'lack of hunger',
    'decrease in food intake', 'disinterest in eating', 'feeling full quickly', 'loss of taste for food', 'sudden loss of appetite', 'absence of hunger', 'nausea with food',
    'difficulty eating', 'reduced food consumption', 'lack of craving for food', 'feeling satiated quickly', 'loss of appetite due to illness', 'loss of appetite from stress',
    'loss of appetite from medication', 'anorexia', 'anorexia nervosa', 'feeling no appetite', 'feeling disinterested in food', 'poor food intake', 'reduced food desire'
],

'constipation': [
    'difficulty passing stool', 'infrequent bowel movements', 'hard stools', 'painful bowel movements', 'feeling of incomplete evacuation', 'straining during bowel movement',
    'constipated', 'dry stool', 'difficulty in defecation', 'delayed bowel movements', 'irregular bowel movements', 'hard and dry stool', 'chronic constipation', 'temporary constipation',                'trouble with bowel movements', 'trouble passing stool', 'slow bowel transit', 'stool retention', 'decreased bowel movement frequency', 'bowel sluggishness',
    'straining to poop', 'bowel movement difficulty', 'slow bowel function', 'lack of bowel movement', 'intestinal irregularity'
],

'flu': [
    'influenza', 'seasonal flu', 'viral flu', 'flu virus', 'common flu', 'flu infection', 'respiratory flu', 'flu-like symptoms', 'flu illness',
    'acute influenza', 'viral infection', 'cold and flu', 'influenza virus', 'flu sickness', 'contagious flu', 'pandemic flu',
    'swine flu', 'avian flu', 'H1N1 flu', 'influenza strain', 'cough and flu', 'flu symptoms', 'body aches from flu', 'cold flu',
    'influenza fever', 'flu epidemic', 'general flu discomfort', 'influenza outbreak', 'flu season illness', 'sudden flu'
],

'infection': [
    'contamination', 'infectious disease', 'germ infection', 'bacterial infection', 'viral infection', 'fungal infection', 'parasite infection', 'microbial infection',
    'pathogen invasion', 'infected area', 'infection outbreak', 'systemic infection', 'local infection', 'wound infection', 'skin infection', 'respiratory infection',
    'urinary tract infection', 'ear infection', 'sinus infection', 'blood infection', 'sepsis', 'foodborne illness', 'infected tissue', 'infection of the bloodstream',
    'infection of the lungs', 'bacterial contamination', 'inflammation from infection', 'infectious agent', 'disease-causing infection', 'contagion', 'infection symptoms',
    'chronic infection', 'acute infection', 'health infection', 'infection spread', 'infection risk', 'infectious condition', 'contagious disease'
],

'inflammation': [
    'inflammatory response', 'immune response', 'chronic inflammation', 'acute inflammation', 'inflammatory reaction', 'inflammation of tissues', 'inflammation in the body',
    'inflammation of the skin', 'joint inflammation', 'internal inflammation', 'inflammatory condition', 'cellular inflammation', 'inflammatory disorder', 'local inflammation',
    'systemic inflammation', 'inflammation from infection', 'inflammation from injury', 'inflammation from disease', 'autoimmune inflammation', 'inflammation in the joints',
    'stiffness from inflammation', 'inflammation from trauma', 'inflammation due to allergies'
],

'cramp': [
    'paining cramp', 'cramped muscle', 'cramping sensation', 'cramping'
],

'bleeding': [
    'blood loss', 'hemorrhage', 'hemorrhaging', 'bloodshed', 'wound bleeding', 'internal bleeding', 'external bleeding', 'bleeding from injury', 'blood flow',
    'spurting blood', 'bleeding wound', 'gushing blood', 'cut bleeding', 'profuse bleeding', 'minor bleeding', 'heavy bleeding', 'excessive bleeding',
    'periodic bleeding', 'bleeding from incision', 'bleeding from surgery', 'abnormal bleeding', 'uncontrolled bleeding', 'bleeding disorder'
],

'irritation': [
    'annoyance', 'chronic irritation', 'temporary irritation', 'allergic irritation', 'irritated feeling'
],

'anxiety': [
    'worry', 'unease', 'stress', 'fear', 'apprehension', 'nervous tension', 'anxiousness', 'nervous anxiety', 'social anxiety', 'generalized anxiety', 'anxiety disorder',   'anticipatory anxiety', 'anxiety attack', 'apprehensive feeling', 'distress', 'emotional unease', 'worrying', 'overthinking', 'mental tension'
],

'depression': [
    'sadness', 'melancholy', 'despair', 'low mood', 'dismay', 'hopelessness', 'discouragement', 'despondency', 'blues', 'dejectedness', 'very sad', 'sad',
    'feeling down', 'feeling hopeless', 'loss of interest', 'unhappiness', 'mental exhaustion', 'loss of joy', 'major depressive disorder',
    'clinical depression', 'chronic depression', 'depressive episode', 'anhedonia', 'negative mood', 'downheartedness'
],

'cancer': [
    'malignant tumor', 'carcinoma', 'neoplasm', 'oncological disease', 'cancerous growth', 'tumor', 'metastatic cancer', 'cancer cells', 'tumor growth',
    'breast cancer', 'lung cancer', 'skin cancer', 'prostate cancer', 'colon cancer', 'leukemia', 'lymphoma', 'sarcoma', 'head and neck cancer',
    'pancreatic cancer', 'bladder cancer', 'stomach cancer', 'cancer diagnosis', 'cancerous tumor', 'fatal cancer', 'chronic cancer', 'advanced cancer',
    'stage 4 cancer', 'cancer treatment', 'chemotherapy', 'radiation therapy', 'cancer stage', 'oncology'
],

'diabetes': [
    'diabetes mellitus', 'high blood sugar', 'high sugar', 'insulin resistance', 'type 1 diabetes', 'type 2 diabetes', 'gestational diabetes', 'sugar diabetes',
    'chronic high blood sugar', 'endocrine disorder', 'metabolic disorder', 'insulin deficiency', 'insulin imbalance', 'glucose intolerance', 'sugar level is high',
    'blood sugar imbalance', 'hyperglycemia', 'diabetic condition', 'diabetic disease', 'diabetic disorder', 'pancreatic disorder', 'non-insulin dependent diabetes',
    'insulin-dependent diabetes', 'pre-diabetes', 'diabetic symptoms', 'diabetic complications', 'diabetes management', 'diabetes treatment'
],

'weight loss': [
    'fat loss', 'loss of body weight', 'slimming down', 'losing pounds', 'weight reduction', 'weight management', 'fat burning', 'weight cut',
    'body slimming', 'reduction in weight', 'fat shedding', 'calorie burning', 'trimming down', 'losing inches', 'dropping weight', 'healthy weight loss',
    'body fat reduction', 'weight loss journey', 'sustainable weight loss', 'rapid weight loss', 'gradual weight loss', 'controlled weight loss',
    'dieting', 'fitness weight loss', 'weight loss goals'
],

'hair loss': [
    'alopecia', 'balding', 'thinning hair', 'hair thinning', 'hair shedding', 'hair fall', 'scalp hair loss', 'bald spots', 'receding hairline',
    'hairline recession', 'hair breakage', 'excessive hair loss', 'temporary hair loss', 'pattern baldness', 'male pattern baldness', 'female pattern baldness',
    'androgenic alopecia', 'patchy hair loss', 'diffuse hair loss', 'hair loss due to stress', 'postpartum hair loss', 'age-related hair loss', 'genetic hair loss',
    'hair fall disorder', 'alopecia areata', 'hair loss condition', 'scalp thinning', 'hair loss treatment'
],

'blurred vision': [
    'vision impairment', 'unclear vision', 'fuzzy vision', 'distorted vision', 'foggy vision', 'hazy vision', 'blurry eyesight', 'impaired vision', 'cannot see properly',
    'vision distortion', 'clouded vision', 'poor vision', 'vision fuzziness', 'difficulty seeing clearly', 'blurred eyesight', 'visual disturbance',
    'unclear eyesight', 'visual impairment', 'blurry sight', 'sight distortion', 'vision problems', 'temporary blurred vision', 'chronic blurred vision',
    'blurry perception'
],

'numbness': [
    'loss of sensation', 'tingling', 'pins and needles', 'lack of feeling', 'reduced sensation', 'sensory loss', 'numb sensation', 'feeling of numbness',
    'numb feeling', 'sensory numbness', 'partial numbness', 'temporary numbness', 'persistent numbness'
],

'dry mouth': [
    'xerostomia', 'cottonmouth', 'parched mouth', 'thirsty mouth', 'dryness in the mouth', 'lack of saliva', 'reduced saliva production', 'mouth dryness',
    'sticky mouth', 'dryness of the oral cavity', 'uncomfortable dry mouth', 'dry tongue', 'thirsty feeling in the mouth', 'saliva deficiency', 'oral dryness',
    'mouth discomfort', 'dryness in the mouth and throat', 'sore dry mouth', 'dehydrated mouth', 'dryness due to medication', 'mouth feels dry', 'no saliva'
],

'frequent urination': [
    'urinary frequency', 'increased urination', 'urinary urgency', 'excessive urination', 'frequent trips to the bathroom', 'overactive bladder',
    'need to urinate often', 'urination urgency', 'recurrent urination', 'constant urination', 'frequent need to pee', 'urgent urination', 'pollakiuria',
    'urinary incontinence', 'nighttime urination', 'nocturia', 'constant need to urinate', 'increased frequency of urination'
],

'acne': [
    'pimples', 'blemishes', 'zits', 'whiteheads', 'blackheads', 'cystic acne', 'teenage acne', 'adult acne', 'pimple outbreaks', 'clogged pores', 'acne vulgaris', 'skin spots', 'face pimples', 'hormonal acne', 'acne lesions', 'acne scars', 'clogged follicles', 'sebaceous gland activity', 'oil acne', 'acne on the back', 'acne on the chest'
],

'difficulty swallowing': [
    'dysphagia', 'trouble swallowing', 'swallowing difficulty', 'painful swallowing', 'difficulty with swallowing', 'difficulty in swallowing food',
    'difficulty swallowing liquids', 'inability to swallow', 'swallowing discomfort', 'choking sensation', 'difficulty swallowing pills', 'food getting stuck',
    'difficulty in throat swallowing', 'swallowing obstruction', 'swallowing problems', 'gagging while swallowing', 'swallowing trouble',
    'feeling of blockage while swallowing'
],

'restlessness': [
    'unease', 'fidgeting', 'inability to relax', 'impatience', 'uneasiness', 'hyperactivity', 'jitteriness', 'inability to stay still', 'unsettledness',
    'fidgety feeling', 'lack of calm', 'shaky feeling'
],

'bloating': [
    'abdominal bloating', 'stomach bloating', 'gas buildup', 'swollen belly', 'feeling of fullness', 'abdominal distention',
    'overfull stomach', 'stomach discomfort', 'stomach swelling', 'intestinal bloating', 'bloated stomach', 'feeling puffed up', 'bloating sensation',
    'gassy stomach', 'stomach pressure', 'bloating after eating', 'digestive bloating', 'feeling bloated', 'bloating in the abdomen', 'gas pain',
    'cramping and bloating'
],

'gas': [
    'flatulence', 'intestinal gas', 'stomach gas', 'abdominal gas', 'gassy feeling', 'farting', 'passing gas', 'gas buildup',
    'wind', 'belching', 'burping', 'gas discomfort', 'gas pains', 'digestive gas', 'stomach is full', 'gassy stomach', 'trapped gas',
    'excessive gas', 'gas expulsion', 'intestinal discomfort', 'gas release', 'unwanted gas', 'gas in the stomach', 'passing wind'
],

'indigestion': [
    'dyspepsia', 'digestive discomfort', 'fullness after eating', 'nausea after eating', 'acidic stomach', 'belching', 'feeling of heaviness', 'difficulty digesting', 'food intolerance', 'excessive burping'
],

'heartburn': [
    'acid reflux', 'gastroesophageal reflux', 'GERD', 'acid indigestion', 'stomach acid', 'burning sensation in the chest', 'burning throat',
    'acidic taste in the mouth', 'chronic heartburn', 'esophageal burning', 'gastric reflux', 'burning chest pain', 'sour stomach', 'acid regurgitation',
    'acid backflow', 'stomach acid reflux', 'acid reflux disease', 'heartburn symptoms', 'irritated esophagus', 'burning sensation after eating',
    'heartburn discomfort'
],

'mouth sore': [
    'oral ulcer', 'canker sore', 'cold sore', 'blister in the mouth', 'mouth ulcer', 'painful mouth lesion', 'sores in the mouth', 'lesions on the gums',
    'painful spot in the mouth', 'mouth blister', 'mouth irritation', 'gum ulcer', 'sore inside the mouth', 'ulcerated mouth tissue', 'painful mouth spot',
    'burning mouth', 'painful tongue spot', 'sores on the lips', 'swollen mouth tissue', 'open mouth wound', 'oral lesion', 'mouth wound', 'infected mouth area'
],

'nosebleed': [
    'epistaxis', 'bleeding from the nose', 'nasal hemorrhage', 'nose bleeding', 'bloody nose', 'hemorrhaging from the nose', 'nose blood flow',
    'spontaneous nosebleed', 'anterior nosebleed', 'posterior nosebleed', 'frequent nosebleeds', 'nosebleed episode', 'bleeding nostrils',
    'blood coming out from the nose', 'nasal bleeding', 'bloody discharge from the nose', 'nasal passage bleeding', 'nosebleed symptoms', 'internal nasal bleeding',
    'nose injury bleeding'
],
'ear ringing': [
    'tinnitus', 'ringing in the ears', 'ear buzzing', 'ear noise', 'persistent ear sound', 'ear whistling', 'ear humming', 'sounds in the ears',
    'ear roaring', 'ringing sound in the ear', 'constant ringing', 'ear congestion', 'noises in the ear', 'buzzing in the ear', 'hissing in the ear',
    'whistling in the ear', 'high-pitched sound', 'low-pitched ear sound', 'phantom sounds', 'ear sensation', 'auditory disturbance'
],

'dark urine': [
    'dark-colored urine', 'dark yellow urine', 'brown urine', 'amber-colored urine', 'tea-colored urine', 'concentrated urine', 'urine with strong color',
    'deep yellow urine', 'urine discoloration', 'darkened urine', 'urine with reddish tint', 'dark brown urine', 'urine with high concentration', 'cloudy urine',
    'urine with abnormal color', 'dark urine caused by medication', 'urine with blood', 'urine with high pigment', 'strong urine color'
],

'blood in urine': [
    'hematuria', 'urinary blood', 'bloody urine', 'red urine', 'urine with blood', 'blood-tinged urine', 'blood in the urine stream', 'pink urine',
    'urinary tract bleeding', 'blood in the urinary tract', 'hemorrhagic urine', 'urinary bleeding', 'presence of blood in urine', 'blood in the bladder',
    'bloody discharge in urine', 'urine with reddish tint', 'blood in the urine sample', 'bleeding from the kidneys', 'blood in the urine after urination',
    'visible blood in urine', 'microscopic hematuria'
],

'blood in stool': [
    'hematochezia', 'rectal bleeding', 'bloody stool', 'stool with blood', 'bright red blood in stool', 'dark blood in stool', 'blood in the bowel movement',
    'blood-tinged stool', 'bloody feces', 'blood in feces', 'stool with reddish tint', 'blood in the stool sample', 'melena', 'dark tarry stool',
    'fecal blood', 'visible blood in stool', 'blood after bowel movement', 'stool with clots', 'bloody discharge from the rectum', 'abnormal stool color'
],

'high blood pressure': [
    'hypertension', 'elevated blood pressure', 'high BP', 'high arterial pressure', 'raised blood pressure', 'increased blood pressure', 'high systolic pressure',
    'high diastolic pressure', 'chronic hypertension', 'hypertensive condition', 'uncontrolled blood pressure', 'borderline hypertension', 'stage 1 hypertension',
    'stage 2 hypertension', 'persistent high blood pressure', 'high blood pressure disorder', 'abnormal blood pressure', 'hypertensive crisis', 'cardiovascular hypertension',
    'risk of hypertension', 'elevated BP', 'hypertensive state', 'BP is high'
],

'low blood pressure': [
    'hypotension', 'low BP', 'decreased blood pressure', 'low arterial pressure', 'reduced blood pressure', 'hypotensive condition', 'low systolic pressure',
    'low diastolic pressure', 'abnormally low blood pressure', 'postural hypotension', 'orthostatic hypotension', 'chronic hypotension', 'mild hypotension',
    'severe hypotension', 'low blood pressure symptoms', 'blood pressure drop', 'low cardiovascular pressure', 'circulatory hypotension', 'inadequate blood pressure',
    'dizzy blood pressure', 'low blood pressure episode', 'BP is low','low blood pressure'
],

'excessive thirst': [
    'polydipsia', 'intense thirst', 'uncontrollable thirst', 'extreme thirst', 'constant thirst', 'increased thirst', 'abnormal thirst', 'drinking more water', 'consuming more water',
    'compulsive thirst', 'thirsty all the time', 'unquenchable thirst', 'chronic thirst', 'intense desire to drink', 'frequent thirst', 'dehydration thirst','thirsty feeling',
    'abnormal fluid intake desire', 'thirst without relief', 'excessive fluid consumption', 'thirst due to dehydration', 'thirsty feeling', 'abnormal hydration needs'
],

'dehydration': [
    'fluid loss', 'water depletion', 'lack of hydration', 'electrolyte imbalance', 'insufficient water intake', 'dehydrating',
    'dehydrated state', 'water deficiency', 'reduced fluid levels', 'severe dehydration', 'mild dehydration', 'dehydration symptoms',
    'fluid imbalance', 'low body water', 'loss of body fluids', 'heat exhaustion', 'low hydration', 'hypohydration'
],

'red eyes': [
    'eye redness', 'bloodshot eyes', 'conjunctival redness', 'inflamed eyes', 'eye irritation', 'eyes with blood vessels',
    'swollen eyes', 'sore eyes', 'tired eyes', 'watery eyes', 'eye inflammation', 'pink eye', 'eye congestion', 'eye discomfort', 'eyes looking inflamed',
    'redness in the eyes', 'burning eyes', 'allergic eyes', 'eyes with a reddish tint'
],

'eye discharge': [
    'ocular discharge', 'eye mucus', 'eye secretion', 'eye crust', 'sticky eyes', 'eye pus', 'yellow eye discharge', 'clear eye discharge', 'water coming out of eyes',
    'green eye discharge', 'gunky eyes', 'eye drainage', 'teary eyes', 'eye secretion buildup', 'crusty eyes', 'eye fluid', 'excessive tear production',
    'morning eye crust', 'sticky eyelids', 'eye infection discharge', 'pus from the eye', 'watery eye discharge', 'eye discharge after sleep',
    'discharge from the tear duct', 'rheum in the eye','something coming out of eyes'
],

'ear discharge': [
    'otorrhea', 'ear fluid', 'ear drainage', 'pus from the ear', 'ear pus', 'ear infection discharge', 'fluid from the ear', 'ear secretion',
    'yellow ear discharge', 'green ear discharge', 'watery ear discharge', 'bloody ear discharge', 'ear mucus', 'crust in the ear', 'excessive ear fluid',
    'ear leakage', 'ear wax buildup', 'discharge from the ear canal', 'discharge from the middle ear', 'infection-related ear discharge', 'ear discharge after swimming',
    'ear drainage after injury', 'something coming out of ears'
],

'balance problem': [
    'vertigo', 'loss of balance', 'balance disorder', 'impaired balance', 'unsteady gait', 'lack of coordination', 'dizziness and unsteadiness',
    'balance difficulty', 'feeling of instability', 'spatial disorientation', 'postural imbalance', 'equilibrium disturbance', 'feeling off-balance',
    'gait imbalance', 'disequilibrium', 'vestibular dysfunction', 'balance issues', 'vertiginous symptoms', 'coordination problems', 'stumbling', 'feeling lightheaded'
],

'irregular heartbeat': [
    'arrhythmia', 'abnormal heartbeat', 'heart palpitations', 'irregular pulse', 'heart rhythm disorder', 'uneven heartbeat', 'skipped heartbeat',
    'rapid heartbeat', 'slow heartbeat', 'tachycardia', 'bradycardia', 'atrial fibrillation', 'ventricular fibrillation', 'heart flutter', 'irregular heart rhythm',
    'heart irregularities', 'palpitations', 'fluttering heart', 'cardiac arrhythmia', 'dysrhythmia', 'irregular pulse rate', 'heartbeat irregularity',
    'irregular heart rate', 'heart pounding'
],

'fainting': [
    'syncope', 'passing out', 'loss of consciousness', 'blackout', 'going unconscious', 'faint', 'collapse', 'temporary unconsciousness',
    'sudden fainting', 'faint spell', 'dizziness and fainting', 'dizzy spell', 'feeling lightheaded', 'near fainting', 'brief loss of consciousness',
    'head rush', 'staggering', 'fainting episode', 'loss of awareness', 'unconsciousness', 'momentary blackout', 'unconscious',
    'dizzy and lightheaded', 'feeling woozy'
],

'nervousness': [
    'nervous tension', 'nervous energy', 'uneasiness', 'nervous feeling', 'worry', 'uneasy feeling', 'jitters', 'nervous anticipation', 'fearfulness', 'shakiness', 'edginess',
    'fidgeting', 'mental unease', 'trepidation', 'feeling on edge', 'worrying', 'nervous butterflies'
],

'panic attack': [
    'anxiety attack', 'nervous breakdown', 'stress attack', 'overwhelming fear', 'intense fear episode', 'fight-or-flight response', 'panic episode', 'emotional breakdown',
    'sudden panic', 'heart-pounding anxiety', 'fear attack', 'panic episode', 'anxiety episode', 'intense panic', 'acute stress response', 'terror attack', 'nervous episode',
    'severe panic', 'acute emotional distress', 'uncontrollable fear', 'chronic panic disorder'
],

'mood swing': [
    'emotional swing', 'mood fluctuation', 'emotional rollercoaster', 'mood shift', 'mood change', 'mood variation', 'mood disorder',
    'rapid mood change', 'emotional instability', 'mood instability', 'mood alteration', 'emotional shift', 'temper fluctuation',
    'emotional lability', 'mood fluctuations', 'unstable mood', 'irregular mood', 'affective swing', 'mood imbalance', 'emotional outbursts',
    'highs and lows', 'emotional extremes'
],

'difficulty concentrating': [
    'inability to focus', 'lack of focus', 'poor concentration', 'trouble focusing', 'concentration problems', 'distractibility',
    'difficulty paying attention', 'lack of mental clarity', 'difficulty staying focused', 'inattention', 'short attention span', 'mind wandering',
    'difficulty concentrating on tasks', 'poor attention span', 'difficulty maintaining focus', 'lack of mental focus', 'difficulty with concentration',
    'easily distracted', 'unable to focus', 'attention issues', 'concentration challenges'
],

'lack of motivation': [
    'demotivated', 'low motivation', 'disinterest', 'lack of drive', 'lack of ambition', 'lack of initiative', 'apathy', 'unmotivated',
    'loss of drive', 'lack of enthusiasm', 'indifference', 'lack of determination', 'lack of purpose', 'loss of interest', 'lack of energy',
    'procrastination', 'lack of willpower', 'lack of focus', 'lack of passion', 'feeling uninspired', 'demotivation', 'lack of commitment', 'indifferent attitude'
],

'exhaustion': [
    'fatigue', 'tiredness', 'weariness', 'drained', 'burnout', 'physical exhaustion', 'mental exhaustion', 'extreme fatigue', 'lack of energy',
    'overwhelming tiredness', 'complete fatigue', 'depletion', 'lack of stamina', 'total exhaustion', 'exhausted feeling', 'chronic fatigue',
    'fatigued state', 'drowsiness', 'wearing out', 'energy depletion', 'fatigue syndrome', 'feeling drained', 'exhaustive tiredness', 'loss of energy',
    'profound fatigue', 'fatigue and weakness'
],

'sprain': [
    'ligament injury', 'joint sprain', 'ligament strain', 'stretched ligament', 'ligament tear',
    'sprained ligament', 'mild sprain', 'severe sprain', 'ligament damage'
],

'strain': [

    'soft tissue strain', 'overexertion', 'overworked muscle', 'acute strain', 'chronic strain', 'ligament strain'
],

'arthritis': [
    'inflammatory arthritis', 'rheumatoid arthritis', 'osteoarthritis', 'degenerative joint disease',  'rheumatism',  'pain from arthritis',
    'arthralgia', 'chronic arthritis', 'autoimmune arthritis', 'psoriatic arthritis'

],

'gout': [
    'uric acid buildup', 'acute gout', 'chronic gout', 'gout attack', 'joint pain from gout', 'gout flare-up', 'gouty inflammation', 'gouty attack',
    'painful gout episode', 'gouty swelling', 'gout in the foot', 'gout in the big toe', 'gouty condition', 'uric acid crystals', 'gouty joint disease'
],

'shoulder pain': [
    'shoulder discomfort', 'pain in the shoulder', 'shoulder ache', 'sharp shoulder pain', 'dull shoulder pain', 'shoulder stiffness', 'rotator cuff pain',
    'shoulder joint pain', 'pain in the shoulder joint', 'shoulder strain', 'shoulder injury', 'shoulder inflammation', 'pain in the upper arm',
    'muscle pain in the shoulder', 'shoulder tenderness', 'pain when moving the shoulder', 'shoulder sprain', 'pain in the deltoid', 'shoulder dislocation pain',
    'pain in the scapula', 'pain after shoulder surgery', 'shoulder impingement pain', 'frozen shoulder', 'pain from shoulder overuse', 'referred shoulder pain'
],

'bone fracture': [
    'broken bone', 'bone break', 'fractured bone', 'cracked bone', 'bone crack', 'bone injury', 'fracture', 'compound fracture', 'closed fracture',
    'stress fracture', 'hairline fracture', 'complete fracture', 'incomplete fracture', 'displaced fracture', 'non-displaced fracture', 'bone splinter',
    'fractured limb', 'fractured bone segment', 'broken limb', 'broken bone segment', 'cracked bone injury', 'bone rupture', 'bone fracture symptoms',
    'fractured bone tissue', 'fracture of the bone', 'crack in bone'
],

'back bone issue': [
    'spinal problem', 'back bone pain', 'spinal condition', 'vertebral issue', 'spinal disorder', 'back injury', 'spinal misalignment', 'back bone is paining',
    'disc herniation', 'sciatica', 'spinal injury', 'neck and back issues', 'spinal discomfort', 'lumbar pain', 'spinal cord issue', 'slipped disc', 'musculoskeletal disorder',
    'spinal health issue', 'postural problems', 'spondylosis', 'degenerative disc disease', 'spine misalignment', 'spinal deformity', 'spinal arthritis', 'spinal stenosis'
],

'female issue': [
    'women’s health', 'gynecological issue', 'female reproductive health', 'menstrual problems', 'menstrual irregularities', 'PCOS', 'endometriosis', 'fibroids', 'ovarian cysts',
    'vaginal infection', 'vaginal discharge', 'fertility issues', 'menopause', 'pre menopause', 'post menopause', 'infertility', 'vaginal dryness', 'prolapsed uterus',
    'birth control issues', 'female urinary issues', 'pregnancy complications','white discharge','female issue','female issues'
],

'thyroid': [
    'thyroid gland', 'hypothyroidism', 'hyperthyroidism', 'thyroid disorder', 'thyroid imbalance', 'underactive thyroid', 'overactive thyroid', 'goiter', 'thyroid dysfunction',
    'thyroid disease', 'thyroid cancer', 'thyroiditis', 'low thyroid function', 'high thyroid function', 'endocrine disorder', 'thyroid nodules', 'thyroid hormone imbalance',
    'TSH imbalance', 'thyroid condition', 'thyroid problems', 'autoimmune thyroid disease', 'pituitary-thyroid dysfunction', 'thyroid health', 'thyroid testing'
],

'piles': [
    'hemorrhoids', 'anal piles', 'rectal swelling', 'swollen veins', 'internal hemorrhoids', 'external hemorrhoids',
    'hemorrhoidal disease', 'rectal discomfort', 'anal itching', 'anal bleeding', 'rectal bleeding', 'chronic hemorrhoids',
    'painful hemorrhoids', 'prolapsed hemorrhoids', 'thrombosed hemorrhoids', 'anal fissures', 'blood clots in hemorrhoids',
    'swollen hemorrhoids', 'anal prolapse', 'inflamed hemorrhoids', 'rectal irritation', 'constipation-related hemorrhoids',
    'itchy anus', 'hemorrhoid treatment', 'hemorrhoid relief'
],

'vomiting': [
'food throwing up', 'puking', 'stomach upset', 'retching', 'nauseated vomiting', 'projectile vomiting', 'forceful expulsion', 'stomach evacuation',
'regurgitation', 'gag reflex', 'vomit episode', 'bilious vomiting', 'dry heaving', 'stomach upheaval', 'continuous vomiting', 'acidic vomit', 'vomiting bile',
'upset stomach leading to vomiting', 'cyclic vomiting', 'food rejection', 'nausea-induced vomiting', 'vomiting spells', 'recurrent vomiting', 'violent vomiting',
'vomiting after eating', 'motion sickness vomiting', 'morning sickness vomiting', 'dehydration from vomiting', 'vomiting blood', 'gastrointestinal vomiting',
'digestive tract expulsion', 'vomiting sensation', 'persistent nausea', 'induced vomiting', 'uncontrollable vomiting', 'abdominal vomiting', 'stomach spasms causing vomiting',
'vomiting reflex', 'vomiting from food poisoning', 'travel sickness vomiting', 'chronic vomiting episodes', 'sudden vomiting', 'intense vomiting', 'frequent retching',
'vomiting with dizziness', 'vomiting due to illness', 'nervous vomiting', 'vomiting from indigestion', 'bitter vomit taste', 'post-vomit weakness', 'vomiting accompanied by sweating'
],

'hearing loss': [
'loss of hearing', 'partial hearing loss', 'complete hearing loss', 'reduced hearing', 'impaired hearing', 'difficulty hearing', 'diminished hearing ability',
'hearing impairment', 'sensorineural hearing loss', 'conductive hearing loss', 'temporary hearing loss', 'permanent hearing loss', 'age-related hearing loss', 'noise-induced hearing loss',
'hearing deficiency', 'blocked hearing', 'muffled hearing', 'ringing in ears', 'ear damage', 'auditory dysfunction', 'ear canal blockage', 'inner ear damage',
'hearing weakness', 'fading hearing', 'loss of sound perception', 'difficulty understanding speech', 'distorted hearing', 'ear drum damage', 'hearing sensitivity reduction',
'unilateral hearing loss', 'bilateral hearing loss', 'gradual hearing loss', 'sudden hearing loss', 'ear infection-related hearing loss', 'fluid in ear causing hearing loss',
'hearing clarity reduction', 'speech comprehension difficulty', 'auditory decline', 'nerve damage causing hearing loss', 'inability to detect sound frequencies', 'ear trauma',
'hearing impairment due to illness', 'hearing degradation', 'low sound perception', 'high-frequency hearing loss', 'earwax blockage hearing loss', 'acoustic trauma',
'temporary auditory loss', 'chronic hearing damage', 'progressive hearing loss'
],

'bone pain': [
'bone tenderness', 'bone swelling', 'aching bones', 'deep bone pain', 'sharp bone pain', 'bone discomfort', 'persistent bone pain', 'localized bone pain',
'throbbing bone sensation', 'bone sensitivity', 'bone ache during movement', 'chronic bone pain', 'bone bruising', 'bone soreness', 'inflammatory bone pain',
'fracture-related bone pain', 'joint and bone pain', 'dull bone ache', 'piercing bone pain', 'bone pain during rest', 'bone pain under pressure', 'bone stiffness',
'osteopathic pain', 'bone fragility pain', 'bone strain', 'bone inflammation', 'bone pain with swelling', 'tender bone surface', 'aching joints and bones',
'deep-seated bone ache', 'bone discomfort while standing', 'bone pain due to injury', 'radiating bone pain', 'stress fracture pain', 'bone pain from infection',
'cancer-related bone pain', 'bone tenderness to touch', 'nighttime bone pain', 'sensitive bone tissue', 'bone marrow pain', 'bone pain during activity',
'osteoporosis-related bone pain', 'bone pain with movement', 'skeletal pain', 'generalized bone pain', 'acute bone discomfort', 'bone stress pain',
'pressure-induced bone pain', 'intense bone ache', 'stiff bone joints', 'localized skeletal pain'
],

'weight gain': [
'increase in weight', 'gain in body mass', 'unintended weight gain', 'gradual weight gain', 'rapid weight gain', 'excess body weight', 'body mass increase',
'weight fluctuation', 'caloric surplus', 'fat accumulation', 'body fat increase', 'muscle mass gain', 'bloating-related weight gain', 'water retention weight gain',
'unhealthy weight gain', 'sudden weight gain', 'weight gain from overeating', 'hormonal weight gain', 'stress-related weight gain', 'weight gain due to inactivity',
'metabolic weight gain', 'post-pregnancy weight gain', 'age-related weight gain', 'diet-induced weight gain', 'weight gain from medication', 'insulin-related weight gain',
'abdominal weight gain', 'upper body weight gain', 'lower body weight gain', 'excess calorie intake', 'fluid retention weight gain', 'chronic weight gain',
'weight gain around the waist', 'poor diet weight gain', 'sedentary lifestyle weight gain', 'hormone imbalance weight gain', 'slow metabolism weight gain',
'unexplained weight gain', 'overeating-induced weight gain', 'fat storage increase', 'weight gain from sugary foods', 'weight gain due to lack of exercise',
'weight gain with bloating', 'hormone-related fat storage', 'weight gain caused by stress eating', 'body composition change', 'progressive weight gain',
'weight gain due to emotional eating', 'weight gain from poor sleep', 'unbalanced diet weight gain','gained weight'
],

'skin burning': [
'burning', 'burning feeling', 'skin irritation', 'skin stinging', 'skin redness', 'skin inflammation', 'burning sensation', 'skin discomfort', 'tingling burn',
'localized skin burn', 'skin heat sensation', 'raw skin feeling', 'skin hypersensitivity', 'sunburn', 'chemical burn', 'skin scorching', 'skin sensitivity to touch',
'prickling skin sensation', 'hot skin feeling', 'burning skin pain', 'skin abrasion burn', 'nerve-related burning', 'itchy burning skin', 'skin damage from burn',
'skin burning after contact', 'intense burning sensation', 'surface skin burn', 'skin blistering', 'persistent skin burn', 'burned skin surface',
'red inflamed skin', 'skin discomfort from heat', 'skin burning from friction', 'skin chafing burn', 'thermal skin burn', 'abrasive skin burn', 'sensitive skin after burn',
'stinging skin pain', 'skin burn from chemicals', 'skin damage sensation', 'skin peeling from burn', 'acute burning feeling', 'skin burn with swelling',
'lingering skin burn', 'burnt skin tenderness', 'skin hot spot', 'irritated skin burn', 'sharp skin burn sensation', 'skin burning rash', 'skin burning itch'
],

'itching': [
'skin itching', 'pruritus', 'itchy sensation', 'persistent itching', 'intense itching', 'localized itching', 'generalized itching', 'skin irritation',
'itchy rash', 'dry skin itching', 'allergic itching', 'itching from insect bites', 'itchy skin patches', 'scalp itching', 'itching sensation under the skin',
'chronic itching', 'temporary itching', 'burning itch', 'itching with redness', 'itching from dryness', 'irritated skin itch', 'tickling skin sensation',
'itchy skin bumps', 'itchy welts', 'itchy hives', 'skin crawling sensation', 'itching from heat rash', 'itchy blisters', 'itching from eczema', 'itching from psoriasis',
'itching with flaking skin', 'itching from fungal infection', 'itching from dermatitis', 'nighttime itching', 'itching due to allergies', 'itching from contact dermatitis',
'itching after sunburn', 'itching from insect stings', 'itching with swelling', 'prickly itching sensation', 'itching from poison ivy', 'itching with dryness',
'itching from skin irritation', 'itching from chemicals', 'itchy mosquito bites', 'itching with inflammation', 'nerve-related itching', 'itching from medication side effects',
'itching due to sweat', 'itching from poor hygiene', 'itchy skin lesions','itching'
],

'injury': [
'injured', 'wound', 'physical injury', 'bodily harm', 'tissue damage', 'acute injury', 'chronic injury', 'sports injury', 'accidental injury',
'soft tissue injury', 'muscle injury', 'joint injury', 'bone injury', 'ligament injury', 'nerve injury', 'head injury', 'spinal injury', 'bruise',
'cut', 'abrasion', 'laceration', 'contusion', 'strain injury', 'sprain injury', 'impact injury', 'blunt force trauma', 'penetrating injury', 'superficial injury',
'deep tissue injury', 'skin injury', 'crush injury', 'internal injury', 'traumatic wound', 'workplace injury', 'repetitive strain injury', 'overuse injury',
'thermal injury', 'chemical injury', 'burn injury', 'frostbite', 'puncture wound', 'tendon injury', 'cartilage injury', 'open wound', 'closed injury',
'localized injury', 'multiple injuries', 'injury with swelling', 'painful injury', 'infected injury'
],

'jaundice': [
'icterus', 'yellow skin', 'yellowing of the eyes', 'skin discoloration', 'yellowish tint', 'bilirubin buildup', 'yellow sclera', 'jaundiced appearance',
'hepatic jaundice', 'neonatal jaundice', 'obstructive jaundice', 'hemolytic jaundice', 'yellowish skin tone', 'liver-related jaundice', 'gallbladder-related jaundice',
'bile duct obstruction', 'elevated bilirubin levels', 'yellow pigmentation', 'pale stools', 'dark urine', 'liver dysfunction symptoms', 'bile buildup', 'yellow facial skin',
'chronic jaundice', 'acute jaundice', 'jaundice-induced fatigue', 'itching with jaundice', 'skin yellowing disorder', 'bile pigment imbalance', 'bilirubin-induced yellowing',
'eye yellowing', 'yellowish mucous membranes', 'jaundice rash', 'jaundice-related weakness', 'jaundice-related nausea', 'jaundice-related abdominal pain', 'hepatitis-related jaundice',
'gallstone-related jaundice', 'pre-hepatic jaundice', 'post-hepatic jaundice', 'yellow palms', 'yellow tongue', 'biliary jaundice', 'jaundice-related weight loss',
'yellow skin patches', 'jaundice-associated itching', 'systemic jaundice', 'yellowish complexion', 'jaundice in newborns', 'jaundice symptoms'
],

'sleepy': [
'sleeping', 'sleepiness', 'drowsy', 'asleep', 'lethargic', 'groggy', 'tired', 'sluggish', 'heavy-eyed', 'nodding off', 'fatigued', 'dozing',
'half asleep', 'sleep-deprived', 'exhausted', 'yawning', 'slow-moving', 'low energy', 'ready for bed', 'snoozy', 'sleep-prone', 'droopy eyed', 'barely awake',
'mentally tired', 'physically tired', 'in need of rest', 'overly relaxed', 'hard to stay awake', 'sleep craving', 'languid', 'wearied', 'brain fog',
'bed ready', 'sleep drawn', 'lazy eyed', 'unfocused from tiredness', 'nodding head', 'drifting off', 'sleepy sensation', 'slumberous', 'soporific', 'somnolent',
'half-awake', 'daytime sleepiness', 'overwhelming fatigue', 'rest-seeking', 'near dozing', 'eyes struggling to stay open', 'unable to concentrate', 'dull from tiredness'
],

'eye weakness': [
'weakness in eyes', 'weak eyes', 'eyes are weak', 'eyes are becoming weak', 'eye is weak', 'tired eyes', 'eye strain', 'blurred vision',
'fatigued eyes', 'heavy eyes', 'difficulty focusing', 'eye muscle weakness', 'strained vision', 'eye discomfort', 'droopy eyelids', 'lack of eye strength',
'vision fatigue', 'eye exhaustion', 'eye tiredness', 'reduced eye stamina', 'difficulty keeping eyes open', 'eye sensitivity', 'failing eye strength',
'eye fatigue after reading', 'poor eye endurance', 'eye weakness from screen use', 'eye weariness', 'visual tiredness', 'eye fragility', 'weak eye muscles',
'prolonged eye strain', 'focus difficulty', 'unstable eye movement', 'eye soreness', 'eye heaviness', 'visual exhaustion', 'difficulty maintaining focus',
'eyes feeling overworked', 'weak visual acuity', 'eye tiredness at night', 'eye discomfort after long tasks', 'loss of eye strength', 'drooping eyes',
'strained eye muscles', 'eye fatigue with headaches', 'difficulty keeping eyes focused', 'eye tension', 'sensitivity to light', 'eye weakness from fatigue'
],


'leg weakness': [
'legs are becoming weak', 'weakness in legs', 'leg is weak', 'legs are weak', 'tired legs', 'wobbly legs', 'unsteady legs', 'shaky legs',
'fatigued legs', 'heavy legs', 'lack of leg strength', 'unstable legs', 'leg muscle weakness', 'difficulty standing', 'difficulty walking', 'leg fatigue',
'legs giving out', 'trembling legs', 'poor leg endurance', 'reduced leg strength', 'weak leg muscles', 'unstable lower limbs', 'leg instability',
'feeling of leg collapse', 'limb weakness', 'leg exhaustion', 'legs feel drained', 'weakness in thigh muscles', 'weakness in calf muscles',
'legs feel powerless', 'numb legs', 'tingling in legs', 'cramping legs', 'difficulty supporting weight', 'legs feel rubbery', 'shaking lower limbs',
'legs feel unstable after exertion', 'difficulty lifting legs', 'leg weakness while climbing stairs', 'leg weakness after standing long',
'legs feel like jelly', 'legs feel heavy and weak', 'poor leg control', 'unstable footing', 'weakness after prolonged standing', 'leg stiffness',
'inability to bear weight on legs', 'sensation of leg failure', 'unresponsive legs'
],

'asthma': ['wheezing', 'reactive airway disease', 'hyperresponsive airway disease', 'asthmatic condition', 'asthmas', 'asthama','whistling sound while breathing'],

'pneumonia': ['lung infection','alveolar infection'],

'sugar': ['sugars', 'glucose', 'blood sugar', 'hyperglycemia', 'hypoglycemia'],
   }

medications_list = [
    "ibuprofen", "acetaminophen", "paracetamol", "aspirin", "naproxen", "acetylsalicylic acid",
    "diclofenac", "meloxicam", "celecoxib", "indomethacin", "ketorolac", "butalbital", "dolo" ,"dolo 650",
    # Antibiotics
    "amoxicillin", "azithromycin", "doxycycline", "ciprofloxacin", "clindamycin", "metronidazole",
    "cephalexin", "amoxicillin-clavulanate", "levofloxacin", "linezolid", "meropenem", "vancomycin",
    "fluconazole", "tetracycline", "rifampin", "sulfamethoxazole-trimethoprim", "nystatin",
    # Cardiovascular
    "lisinopril", "atorvastatin", "simvastatin", "rosuvastatin", "furosemide", "clopidogrel", "warfarin",
    "heparin", "digoxin", "nifedipine", "amlodipine", "atenolol", "tamsulosin", "isosorbide mononitrate",
    "losartan", "metoprolol", "propranolol", "hydralazine", "captopril", "carvedilol", "valsartan",
    "eplerenone", "nicorandil", "ranolazine", "dobutamine", "nitrate", "statin",
    # Diabetes and Endocrine
    "metformin", "insulin", "glimepiride", "glipizide", "glyburide", "sitagliptin", "pioglitazone",
    "liraglutide", "exenatide", "liraglutide", "canagliflozin", "dapagliflozin", "levothyroxine",
    "sodium bicarbonate", "hydrocortisone", "prednisone", "levothyroxine", "desmopressin", "teriparatide",
    # Respiratory / Allergies
    "albuterol", "cetirizine", "loratadine", "fexofenadine", "salbutamol", "montelukast", "levocetirizine",
    "betahistine", "fluticasone", "budesonide", "beclometasone", "theophylline", "ipratropium", "mometasone",
    "salmeterol", "formoterol", "tuscalo", "aminophylline", "naloxone", "dextromethorphan",
    # Mental Health / Neurology
    "sertraline", "citalopram", "escitalopram", "gabapentin", "hydrocodone", "codeine", "tramadol",
    "lorazepam", "diazepam", "clonazepam", "melatonin", "antidepressant", "antianxiety", "clonidine", "aripiprazole",
    "quetiapine", "risperidone", "lithium", "vortioxetine", "duloxetine", "venlafaxine", "bupropion", "mirtazapine",
    "buspirone", "modafinil", "carbamazepine", "topiramate", "lamotrigine", "valproic acid", "levetiracetam",
    "phenytoin", "oxcarbazepine", "zaleplon", "zolpidem", "eszopiclone",
    # Gastrointestinal
    "omeprazole", "pantoprazole", "antacid", "metoclopramide", "ranitidine", "famotidine", "esomeprazole",
    "lansoprazole", "hydrochlorothiazide", "bismuth subsalicylate", "bisacodyl", "docusate", "loperamide", "diphenoxylate",
    "aluminum hydroxide", "sucralfate", "misoprostol", "codeine phosphate",
    # Vitamins and Supplements
    "zinc", "vitamin c", "vitamin d", "multivitamin", "folic acid", "vitamin b12", "vitamin e",
    "iron sulfate", "calcium carbonate", "magnesium oxide", "potassium chloride", "manganese", "iodine",
    "biotin", "collagen", "probiotic", "omega-3", "fish oil",
    # Other
    "prednisone", "hydroxychloroquine", "betahistine", "antibiotic", "antiplatelet", "anticoagulant", "fentanyl",
    "methadone", "buprenorphine", "naloxone", "acetylcysteine", "digoxin", "thiamine", "fluphenazine",
    "morphine", "methocarbamol", "colchicine", "dantrolene", "loperamide", "theophylline", "apixaban",
    "dabigatran", "rivaroxaban", "pantoprazole", "benzonatate", "immunoglobulin", "neostigmine", "levodopa",
    "entacapone", "bromocriptine", "carbidopa", "tizanidine", "probenecid", "allopurinol", "febuxostat",
    "colchicine", "methylprednisolone", "hydrocortisone", "bupropion", "clozapine", "chlorpromazine", "phenelzine",
    "tranylcypromine", "bromocriptine", "tretinoin", "hydroxyzine", "terbinafine", "dapsone", "lidocaine",
    "hydroxyurea", "azathioprine", "cyclophosphamide", "methotrexate", "sulfasalazine", "dimercaprol",
    "dantrolene", "adrenaline", "epinephrine", "tylenol", "xanax", "valium", "ambien", "ativan", "prozac",
    # Dermatology (Skin-related)
    "clotrimazole", "ketoconazole", "hydrocortisone", "tretinoin", "benzoyl peroxide", "clindamycin gel",
    "adapalene", "hydroxychloroquine", "salicylic acid", "calcipotriene", "betamethasone", "mupirocin",
    "fluticasone cream", "permethrin", "coal tar", "finasteride", "isotretinoin", "gentamicin", "pimecrolimus",
    "tacrolimus", "azelaic acid",
    # Musculoskeletal Disorders
    "methocarbamol", "baclofen", "tizanidine", "carisoprodol", "cyclobenzaprine", "diclofenac gel",
    "celecoxib", "indomethacin", "colchicine", "allopurinol", "febuxostat", "naproxen", "hydroxychloroquine",
    "prednisone", "methylprednisolone", "gabapentin", "tramadol", "etoricoxib", "oxcarbazepine", "tizanidine",
    # Immunosuppressants / Immunology
    "methotrexate", "azathioprine", "mycophenolate mofetil", "cyclophosphamide", "tacrolimus", "sirolimus",
    "prednisolone", "bendamustine", "rituximab", "infliximab", "adalimumab", "etanercept", "leflunomide",
    "hydroxychloroquine", "intravenous immunoglobulin", "tofacitinib", "abatacept", "etanercept", "sulfasalazine",
    "cyclosporine", "belimumab",
    # Ophthalmology (Eye-related)
    "latanoprost", "timolol", "brimonidine", "prednisolone acetate", "neomycin-polymyxin-bacitracin", "pilocarpine",
    "dorzolamide", "bimatoprost", "hydroxychloroquine", "tobramycin", "moxifloxacin", "gentamicin drops",
    "sulfacetamide", "cyclopentolate", "hydrocortisone eye drops", "levofloxacin ophthalmic", "ketorolac ophthalmic",
    "flurbiprofen", "povidone-iodine", "dexamethasone eye drops", "lisinopril", "azithromycin eye drops",
    "acetazolamide (for glaucoma)",
    # Urology
    "finasteride", "tamsulosin", "sildenafil", "terazosin", "dutasteride", "vardenafil", "tadalafil", "alfuzosin",
    "oxybutynin", "tolterodine", "mirabegron", "desmopressin", "bethanechol", "dapoxetine", "flomax", "proscar",
    "silodosin", "bupropion", "indomethacin", "methyltestosterone", "tadalafil", "hydroxyurea", "gabapentin",
    "tramadol", "famotidine", "alpha blockers", "calcitonin", "hydrocodone", "morphine",
    # Anti-Viral and Anti-fungal
    "oseltamivir", "zanamivir", "acylovir", "valacyclovir", "ganciclovir", "amphotericin", "fluconazole",
    "terbinafine", "itraconazole", "nystatin", "clotrimazole", "ketoconazole", "miconazole", "fluconazole", "terbinafine",
    "griseofulvin", "miconazole", "nystatin", "terbinafine"
]

location_list = [
    "Mumbai", "Delhi", "Bangalore", "Hyderabad", "Kolkata","Jaipur"
]

# Define symptom_keywords to look for
symptom_keywords = ['pain', 'discomfort', 'ache', 'sore', 'burning', 'itching', 'tingling', 'numbness', 'trouble']

body_parts = [
    'leg','legs', 'eye','eyes', 'hand','hands', 'arm','arms', 'head', 'back', 'chest', 'wrist','wrists', 'throat', 'stomach',
    'neck', 'knee','knees', 'foot','foots', 'shoulder', 'shoulders', 'ear', 'ears','nail' , 'nails', 'bone','bones', 'joint','joints', 'skin','abdomen',
    'mouth', 'nose', 'tooth', 'tooths', 'tongue','lip', 'lips', 'cheek','cheeks', 'chin', 'forehead',
    'elbow', 'elbows','ankle','ankles', 'heel', 'heels', 'toe', 'toes','finger','fingers', 'thumb', 'thumbs', 'palm','palms', 'soles',
    'fingertip', 'fingertips', 'instep', 'calf', 'shin','lumbar', 'thoracic', 'cervical', 'gastrointestinal', 'abdominal', 'rectal', 'genital',
    'urinary', 'respiratory', 'cardiac', 'pulmonary', 'digestive', 'cranial', 'facial',
    'ocular', 'otologic', 'nasal', 'oral', 'buccal', 'lingual', 'pharyngeal', 'laryngeal',
    'trigeminal', 'spinal', 'peripheral', 'visceral', 'biliary', 'renal', 'hepatic'
]

body_parts = list(set(body_parts))  # Ensure uniqueness

candidates = []

# Add symptoms from the symptom list
for symptom in symptom_list:
    candidates.append({"text": symptom, "type": "SYMPTOM", "canonical": symptom})

# Add manual symptom synonyms
for canonical, synonyms in symptom_synonyms.items():
    for synonym in synonyms:
        candidates.append({"text": synonym, "type": "SYMPTOM", "canonical": canonical})

# Generate combinations of body parts and symptom keywords
for body_part in body_parts:
    for keyword in symptom_keywords:
        # Singular and plural combinations
        candidates.append({
            "text": f"{body_part} {keyword}",
            "type": "SYMPTOM",
            "canonical": f"{body_part} pain" if keyword in ['pain', 'ache'] else f"{body_part} {keyword}"
        })

# ----------------------------
# Encode Candidates with SBERT
# ----------------------------
model = SentenceTransformer("all-MiniLM-L6-v2")
candidate_texts = [cand["text"] for cand in candidates]
print(f"Encoding {len(candidate_texts)} candidate phrases...")
embeddings = model.encode(candidate_texts, convert_to_numpy=True)

# Normalize embeddings for cosine similarity
faiss.normalize_L2(embeddings)

# ----------------------------
# Build FAISS Index
# ----------------------------
embedding_dim = embeddings.shape[1]
index = faiss.IndexFlatIP(embedding_dim)
index.add(embeddings)
print(f"FAISS index built with {index.ntotal} candidates.")

# ----------------------------
# Save Index and Candidates
# ----------------------------
faiss.write_index(index, "faiss_index.index")
with open("candidates.json", "w", encoding="utf8") as f:
    json.dump(candidates, f, indent=2)

print("Index and candidates saved successfully.")

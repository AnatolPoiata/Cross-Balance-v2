"""Results-independent text of the article (EN / RO): title block,
introduction, related research, method and experimental design.
Inline markup: *italic*, **bold**, _{sub}, ^{sup} (see docx_builder.add_runs).
"""

TITLE = {
    "en": "CROSS-BALANCE: A SELF-PACED UNDERSAMPLING ENSEMBLE FOR IMBALANCED CLASSIFICATION "
          "IN FRAUD AND CREDIT RISK SCREENING",
    "ro": "CROSS-BALANCE: UN ANSAMBLU CU SUBEȘANTIONARE AUTO-RITMATĂ PENTRU CLASIFICAREA "
          "DEZECHILIBRATĂ ÎN DETECTAREA FRAUDEI ȘI EVALUAREA RISCULUI DE CREDIT",
}

AUTHORS = {
    "en": ["Anatol POIATA", "Academy of Economic Studies of Moldova, Chisinau, Republic of Moldova",
           "anatol.poiata@gmail.com"],
    "ro": ["Anatol POIATA", "Academia de Studii Economice a Moldovei, Chișinău, Republica Moldova",
           "anatol.poiata@gmail.com"],
}

KEYWORDS = {
    "en": "cost-sensitive evaluation; ensemble learning; fraud detection; imbalanced classification; "
          "undersampling",
    "ro": "clasificare dezechilibrată; detectarea fraudei; evaluare sensibilă la costuri; "
          "învățare prin ansamblu; subeșantionare",
}

JEL = "C38; C45; C55; G21"

LABEL = {
    "en": dict(abstract="Abstract", keywords="Key words", jel="JEL Classification", table="Table",
               figure="Figure"),
    "ro": dict(abstract="Rezumat", keywords="Cuvinte-cheie", jel="Clasificare JEL", table="Tabelul",
               figure="Figura"),
}


def intro(L):
    if L == "en":
        return [
            ("h", "I. INTRODUCTION"),
            ("p", "Fraud detection and credit-risk screening are rare-event problems: a bank may see one "
                  "fraudulent card payment in several hundred, or one default in five to ten loans. A model "
                  "that ignores the minority class can still reach high accuracy, so imbalance-aware methods "
                  "should be judged by how well they rank rare cases and by what that ranking costs and saves "
                  "once an institution acts on it (Elkan, 2001; Saito and Rehmsmeier, 2015; Dal Pozzolo, "
                  "Caelen et al., 2014)."),
            ("p", "Undersampling ensembles are a natural answer to this problem. Each member sees all "
                  "minority cases and a different sample of the majority class, so no information is "
                  "discarded overall and every member trains on a balanced problem (Liu, Wu et al., 2009). "
                  "Two weaknesses limit their practical use. If later samples are steered towards "
                  "misclassified majority cases with unbounded weights, a few noisy cases can dominate them "
                  "(Liu, Cao et al., 2020a). Members trained at an artificial prevalence return scores that "
                  "overstate the minority probability, so cost-based decisions go wrong (Dal Pozzolo, Caelen et "
                  "al., 2015). A credible evaluation must also compare such ensembles with gradient-boosted "
                  "trees, which dominate tabular benchmarks (Grinsztajn, Oyallon et al., 2022), on data that "
                  "played no part in their design."),
            ("p", "This paper proposes Cross-Balance, an undersampling ensemble that addresses these issues. Its "
                  "chunks are drawn by self-paced hardness sampling on the ensemble so far, which emphasises "
                  "hard majority cases without letting them take over. Its members alternate gradient-boosting "
                  "and extremely-randomised-tree learners. A prior correction compensates for the probability "
                  "distortion caused by undersampling. The configuration is selected on a predefined subset of "
                  "the development data and then evaluated once on 17 untouched hold-out datasets against ten "
                  "default-configured baselines, including XGBoost, LightGBM, "
                  "RUSBoost and Self-paced Ensemble. The evaluation uses repeated cross-validation, the "
                  "Friedman, Nemenyi and Wilcoxon-Holm tests, an ablation of every component, and a cost "
                  "analysis on European card transactions that links ranking quality to investigation budgets "
                  "and fraud losses."),
        ]
    return [
        ("h", "I. INTRODUCERE"),
        ("p", "Detectarea fraudei și evaluarea riscului de credit sunt probleme cu evenimente rare: o bancă "
              "poate observa o plată frauduloasă la câteva sute de tranzacții sau o neplată la cinci-zece "
              "credite. Un model care ignoră clasa minoritară poate avea totuși o acuratețe ridicată; de aceea "
              "metodele pentru date dezechilibrate trebuie judecate după cât de bine ordonează cazurile rare "
              "și după costurile și economiile pe care le produce această ordonare atunci când instituția "
              "acționează pe baza ei (Elkan, 2001; Saito and Rehmsmeier, 2015; Dal Pozzolo, Caelen et al., 2014)."),
        ("p", "Ansamblurile cu subeșantionare sunt un răspuns firesc la această problemă: fiecare membru vede "
              "toate cazurile minoritare și un alt eșantion din clasa majoritară, astfel încât, per ansamblu, "
              "nicio informație nu se pierde, iar fiecare membru învață o problemă echilibrată (Liu, Wu et al., "
              "2009). Două slăbiciuni le limitează însă utilizarea practică. Dacă eșantioanele ulterioare sunt "
              "orientate spre cazurile majoritare greșit clasificate prin ponderi nelimitate, câteva cazuri "
              "zgomotoase le pot domina (Liu, Cao et al., 2020a). Membrii antrenați la o prevalență artificială "
              "produc scoruri care supraestimează probabilitatea clasei minoritare, iar deciziile bazate pe "
              "costuri devin eronate (Dal Pozzolo, Caelen et al., 2015). În plus, o evaluare credibilă trebuie "
              "să compare astfel de ansambluri cu arborii gradient boosting, care domină comparațiile pe date "
              "tabelare (Grinsztajn, Oyallon et al., 2022), pe date care nu au contribuit la proiectarea lor."),
        ("p", "Articolul propune Cross-Balance, un ansamblu cu subeșantionare care tratează aceste "
              "probleme. Eșantioanele sunt extrase prin eșantionare auto-ritmată după dificultatea estimată de "
              "ansamblul curent, care accentuează cazurile majoritare dificile fără a le permite să domine; "
              "membrii alternează modele gradient boosting și arbori extrem randomizați; o corecție a priori "
              "compensează distorsionarea probabilităților produsă de subeșantionare. Configurația este aleasă "
              "pe o submulțime predefinită a datelor de dezvoltare și evaluată o singură dată pe 17 seturi de "
              "control neatinse, față de zece metode de referință cu setări implicite, "
              "inclusiv XGBoost, LightGBM, RUSBoost și Self-paced Ensemble. Evaluarea folosește validare "
              "încrucișată repetată, testele Friedman, Nemenyi și Wilcoxon-Holm, o ablație a fiecărei "
              "componente și o analiză de costuri pe tranzacții cu carduri europene, care leagă calitatea "
              "ordonării de bugetul de investigare și de pierderile din fraudă."),
    ]


def related(L):
    if L == "en":
        return [
            ("h", "II. RELATED RESEARCH"),
            ("p", "Surveys group imbalance remedies into data-level resampling, algorithm-level cost "
                  "weighting and ensembles (He and Garcia, 2009; Krawczyk, 2016; Fernández, García et al., "
                  "2018). SMOTE and its variants interpolate synthetic minority cases (Chawla, Bowyer et al., "
                  "2002), and class weights scale the loss of each error type (Elkan, 2001). Ensembles combine "
                  "resampling with variance reduction. Balanced Random Forest draws a balanced bootstrap for "
                  "every tree (Chen, Liaw et al., 2004), EasyEnsemble and BalanceCascade train on many majority "
                  "subsets (Liu, Wu et al., 2009), and RUSBoost embeds random undersampling in AdaBoost "
                  "(Seiffert, Khoshgoftaar et al., 2010)."),
            ("p", "The closest relative of Cross-Balance is Self-paced Ensemble (SPE), which undersamples the "
                  "majority class by the hardness that the current ensemble assigns to each case, harmonising the "
                  "contribution of easy, borderline and noisy regions (Liu, Cao et al., 2020a). MESA learns "
                  "this sampling policy by meta-learning (Liu, Wei et al., 2020b). Partition-based hybrids that "
                  "split and pair data regions before voting have recently been proposed as well (Theephoowiang "
                  "and Hanskunatai, 2025). On tabular data, gradient-boosted trees remain hard to beat (Chen and "
                  "Guestrin, 2016; Ke, Meng et al., 2017; Grinsztajn, Oyallon et al., 2022), so any new "
                  "imbalance method has to be compared with weighted boosting, not only with resampled linear "
                  "models."),
            ("p", "Undersampling shifts the class prior seen by the learner, so its scores overstate the "
                  "minority probability. Dal Pozzolo, Caelen et al. (2015) derive the correction used below, "
                  "while Niculescu-Mizil and Caruana (2005) compare general calibration methods. For "
                  "evaluation, precision-recall summaries are preferred to ROC at low prevalence (Saito and "
                  "Rehmsmeier, 2015), and comparisons over many datasets should use rank-based tests "
                  "(Demšar, 2006). In fraud detection, example-dependent costs, in which a missed fraud costs its "
                  "amount and an alert costs an investigation, link model output to money (Correa Bahnsen, "
                  "Aouada et al., 2016; Hernández Aros, Bustamante Molano et al., 2024)."),
        ]
    return [
        ("h", "II. CERCETĂRI CONEXE"),
        ("p", "Sintezele de specialitate grupează soluțiile pentru dezechilibru în reeșantionare, ponderarea "
              "costurilor la nivel de algoritm și ansambluri (He and Garcia, 2009; Krawczyk, 2016; Fernández, "
              "García et al., 2018). SMOTE și variantele sale generează observații minoritare sintetice prin "
              "interpolare (Chawla, Bowyer et al., 2002), iar ponderile de clasă scalează costul fiecărui tip de "
              "eroare (Elkan, 2001). Ansamblurile combină reeșantionarea cu reducerea varianței: Balanced Random "
              "Forest extrage un eșantion echilibrat pentru fiecare arbore (Chen, Liaw et al., 2004), "
              "EasyEnsemble și BalanceCascade învață pe multe submulțimi majoritare (Liu, Wu et al., 2009), iar "
              "RUSBoost integrează subeșantionarea aleatoare în AdaBoost (Seiffert, Khoshgoftaar et al., 2010)."),
        ("p", "Cea mai apropiată metodă de Cross-Balance este Self-paced Ensemble (SPE), care subeșantionează "
              "clasa majoritară după dificultatea atribuită fiecărui caz de ansamblul curent, echilibrând "
              "contribuția regiunilor ușoare, de frontieră și zgomotoase (Liu, Cao et al., 2020a); MESA învață "
              "această politică prin meta-învățare (Liu, Wei et al., 2020b). Recent au fost propuși și algoritmi "
              "hibrizi care partiționează și împerechează regiuni ale datelor înainte de vot (Theephoowiang and "
              "Hanskunatai, 2025). Pe date tabelare, arborii cu gradient boosting rămân greu de depășit (Chen and "
              "Guestrin, 2016; Ke, Meng et al., 2017; Grinsztajn, Oyallon et al., 2022); prin urmare, o metodă "
              "nouă trebuie comparată cu boosting ponderat, nu doar cu modele liniare reeșantionate."),
        ("p", "Subeșantionarea modifică probabilitatea a priori văzută de model, astfel încât scorurile "
              "supraestimează probabilitatea clasei minoritare; Dal Pozzolo, Caelen et al. (2015) derivă "
              "corecția folosită mai jos, iar Niculescu-Mizil and Caruana (2005) compară metode generale de "
              "calibrare. La prevalență scăzută, sintezele precizie-sensibilitate sunt preferabile curbei ROC "
              "(Saito and Rehmsmeier, 2015), iar comparațiile pe multe seturi de date trebuie să folosească teste "
              "bazate pe ranguri (Demšar, 2006). În detectarea fraudei, costurile dependente de observație (o "
              "fraudă ratată costă suma ei, o alertă costă o investigație) leagă ieșirea modelului de bani "
              "(Correa Bahnsen, Aouada et al., 2016; Hernández Aros, Bustamante Molano et al., 2024)."),
    ]


def method(L, m_frac):
    """m_frac: docx_builder.m_frac, passed in to build OMML fractions.
    The first two blocks (heading + overview) are followed by Table 1 in the build script."""
    en = L == "en"
    return [
        ("h", "III. THE CROSS-BALANCE METHOD" if en else "III. METODA CROSS-BALANCE"),
        ("p", "Cross-Balance trains K members in sequence. Every member sees a sample of the minority class "
              "together with a sample of the majority class drawn according to how hard the members trained so far "
              "find each majority case. The member scores are averaged and mapped back to the training prior. "
              "Table 1 summarises each design choice together with the alternative examined in the ablation "
              "(Section V)."
         if en else
         "Cross-Balance antrenează succesiv K membri. Fiecare membru vede un eșantion din clasa minoritară și "
         "un eșantion din clasa majoritară, extras în funcție de cât de dificil este fiecare caz majoritar "
         "pentru membrii antrenați deja. Scorurile membrilor se mediază și se readuc la probabilitatea a priori "
         "a datelor de antrenare. Tabelul 1 rezumă fiecare alegere de proiectare împreună cu alternativa "
         "examinată în ablație (Secțiunea V)."),
        ("sub", "Chunk construction" if en else "Construirea eșantioanelor"),
        ("p", "Let n_{min} and n_{maj} be the class counts of the training data, r the majority-to-minority "
              "ratio inside a chunk and C a cap on chunk size. Each of the K chunks contains m_{maj} majority and "
              "m_{min} minority cases:"
         if en else
         "Fie n_{min} și n_{maj} numărul de observații din cele două clase, r raportul majoritar-minoritar "
         "dintr-un eșantion și C plafonul dimensiunii acestuia. Fiecare dintre cele K eșantioane conține m_{maj} "
         "observații majoritare și m_{min} minoritare:"),
        ("eq", ["m_{maj} = min(r n_{min}, n_{maj}),   m_{min} = n_{min}   if  n_{min} + m_{maj} ≤ C"], 1),
        ("p", "Above the cap, both counts are scaled down in the same proportion, preserving r. The defaults "
              "are K = 50, r = 4 and C = 3,000. They were chosen on development data only (Appendix A)."
         if en else
         "Peste plafon, ambele numere se reduc proporțional, păstrând r. Valorile implicite sunt K = 50, "
         "r = 4 și C = 3.000, alese exclusiv pe datele de dezvoltare (Anexa A)."),
        ("sub", "Self-paced hardness sampling" if en else "Eșantionarea auto-ritmată după dificultate"),
        ("p", "After member t, every majority case x has a hardness h_{t}(x), equal to the mean probability of "
              "the minority class that members 1 to t assign to it. Majority cases are split into B = 5 "
              "equal-width hardness bins. Bin b, with mean hardness h̄_{b}, receives a share of the next "
              "chunk proportional to"
         if en else
         "După membrul t, fiecare observație majoritară x are o dificultate h_{t}(x), egală cu media "
         "probabilităților clasei minoritare atribuite de membrii 1…t. Observațiile majoritare sunt împărțite "
         "în B = 5 intervale egale de dificultate; intervalul b, cu dificultatea medie h̄_{b}, primește din "
         "următorul eșantion o cotă proporțională cu"),
        ("eq", ["q_{b} ∝ ", m_frac(["1"], ["h̄_{b} + α_{t}"]), ",   α_{t} = tan(", m_frac(["π t"], ["2(K − 1)"]),
                ")"], 2),
        ("p", "Cases are then drawn without replacement inside each bin; if a bin's quota exceeds the number of "
              "cases it holds, that bin alone is sampled with replacement (Liu, Cao et al., 2020a). Early members "
              "(α near 0) see easy and hard regions in balance. As α_{t} grows the quotas become equal per "
              "bin, so hard cases are emphasised but never take over a chunk. This matters because the hardest "
              "majority cases may be mislabelled or overlap with the minority class, and error-driven schemes "
              "that keep multiplying their weights can fill late chunks with repeated copies of such cases "
              "(Liu, Cao et al., 2020a). Unlike SPE, which uses one "
              "unpruned tree per subset and ratio 1, Cross-Balance alternates a histogram gradient-boosting "
              "model (50 iterations, 15 leaves) with an extremely-randomised-trees forest of 20 trees (Geurts, "
              "Ernst et al., 2006). Mixing sequential boosting with randomised averaging is intended to diversify "
              "the members; the ablation (Table 6) shows the gain from the mix, although the correlation of "
              "member errors was not measured directly. The larger ratio r = 4 lets every member see more of "
              "the majority distribution."
         if en else
         "Observațiile se extrag apoi fără întoarcere în interiorul fiecărui interval; dacă cota unui interval "
         "depășește numărul de cazuri din el, doar acel interval este eșantionat cu întoarcere (Liu, Cao et al., "
         "2020a). "
         "Primii membri (α aproape de 0) văd echilibrat regiunile ușoare și dificile; pe măsură ce α_{t} crește, "
         "cotele devin egale pe intervale, astfel încât cazurile dificile sunt accentuate fără a putea domina "
         "un eșantion. Acest lucru contează, deoarece cele mai dificile cazuri majoritare pot fi etichetate greșit "
         "sau se pot suprapune cu clasa minoritară, iar schemele bazate pe erori care își multiplică nelimitat "
         "ponderile pot umple ultimele eșantioane cu copii ale acestor cazuri (Liu, Cao et al., 2020a). Spre deosebire de SPE (un arbore neretezat per submulțime, raport 1), Cross-Balance "
         "alternează un model gradient boosting pe histograme (50 de iterații, 15 frunze) cu o pădure de 20 de "
         "arbori extrem randomizați (Geurts, Ernst et al., 2006). Combinarea boosting-ului secvențial cu medierea randomizată urmărește "
         "diversificarea membrilor; ablația (Tabelul 6) arată câștigul acestei combinații, deși corelația "
         "erorilor membrilor nu a fost măsurată direct. Raportul r = 4 permite fiecărui membru să vadă mai mult "
         "din distribuția majoritară."),
        ("sub", "Aggregation, prior correction and decisions" if en else
         "Agregarea, corecția a priori și deciziile"),
        ("p", "The ensemble score s(x) is the mean of the members' minority-class scores. Because a chunk "
              "keeps a fraction β = m_{maj}/n_{maj} of the majority and γ = m_{min}/n_{min} of the minority, "
              "the score is mapped back to the training prior (Dal Pozzolo, Caelen et al., 2015):"
         if en else
         "Scorul ansamblului s(x) este media scorurilor membrilor pentru clasa minoritară. Deoarece un "
         "eșantion păstrează fracțiunea β = m_{maj}/n_{maj} din clasa majoritară și γ = m_{min}/n_{min} din "
         "cea minoritară, scorul este readus la probabilitatea a priori a datelor de antrenare (Dal Pozzolo, "
         "Caelen et al., 2015):"),
        ("eq", ["p(x) = ", m_frac(["k s(x)"], ["k s(x) − s(x) + 1"]), ",   k = β / γ"], 3),
        ("p", "The mapping is monotone, so it leaves every ranking measure unchanged, but it makes p(x) usable "
              "in a cost rule. With transaction amount A(x) and investigation cost c, a case is flagged when "
              "the expected recovered loss exceeds the cost of reviewing it (Elkan, 2001; Correa Bahnsen, "
              "Aouada et al., 2016):"
         if en else
         "Transformarea este monotonă, deci nu modifică nicio măsură de ordonare, dar face ca p(x) să poată fi "
         "folosită într-o regulă de cost: cu suma tranzacției A(x) și costul investigației c, un caz este "
         "semnalat atunci când pierderea așteptată recuperată depășește costul verificării (Elkan, 2001; "
         "Correa Bahnsen, Aouada et al., 2016):"),
        ("eq", ["flag(x)  ⇔  p(x) A(x) > c"], 4),
        ("p", "For a single operating point, predict() can instead use an F1-optimal threshold estimated from "
              "internal three-fold out-of-fold scores. The method is implemented as a scikit-learn-compatible "
              "estimator. The code, data loaders and result files are available from the author."
         if en else
         "Pentru un singur punct de operare, metoda predict() poate folosi în schimb un prag optim pentru F1, "
         "estimat din scoruri interne out-of-fold pe trei partiții. Metoda este implementată ca estimator "
         "compatibil scikit-learn; codul, funcțiile de încărcare a datelor și fișierele de rezultate sunt "
         "disponibile la autor."),
    ]


def design(L):
    en = L == "en"
    return [
        ("h", "IV. EXPERIMENTAL DESIGN" if en else "IV. DESIGNUL EXPERIMENTAL"),
        ("p", "**Data.** Thirty-two binary datasets are split into 15 development and 17 hold-out datasets "
              "(Table 2). The development group comprises three synthetic sets generated with scikit-learn "
              "(Pedregosa, Varoquaux et al., 2011) and label noise, simulated card fraud (Sparkov), PaySim "
              "mobile-money fraud (Lopez-Rojas, Elmir et al., 2016), credit default (Yeh and Lien, 2009), bank "
              "telemarketing (Moro, Cortez et al., 2014), Communities & Crime (Redmond and Baveja, 2002), "
              "Mammographic mass (Elter, Schulz-Wendtland et al., 2007), Secondary mushroom (Wagner, Heider et "
              "al., 2021), Ailerons, two KEEL sets (Alcalá-Fdez, Fernández et al., 2011) and two Kaggle "
              "transaction sets; sources are listed under Table 2. Design and hyper-parameter selection used a "
              "predefined subset of 10 of the 15 development datasets (Appendix A). Four development sets are "
              "not imbalanced (32-50% minority). They are kept as controls that show whether the method "
              "degrades when imbalance is mild; one of them, Mammographic mass, belongs to the selection "
              "subset. The 17 hold-out datasets were not loaded until the configuration was fixed. They comprise the ULB "
              "credit-card fraud data, 284,807 European transactions with 492 frauds (Dal Pozzolo, Caelen et "
              "al., 2015), which is run at full size and native prevalence, and the 16 KEEL- and UCI-derived "
              "benchmark sets compiled by Park and Im (2024) for Park, Lee et al. (2024), with imbalance "
              "ratios from 10 to 42. Development sets above 50,000 rows are stratified-subsampled with at "
              "least 200 positives, which raises PaySim's prevalence from 0.13% to 0.40%."
         if en else
         "**Date.** Cele 32 de seturi binare sunt împărțite în 15 seturi de dezvoltare și 17 seturi de "
         "control (Tabelul 2). Grupul de dezvoltare cuprinde trei seturi sintetice generate cu scikit-learn "
         "(Pedregosa, Varoquaux et al., 2011) și zgomot de etichetare, fraudă simulată cu carduri (Sparkov), "
         "frauda PaySim în plăți mobile (Lopez-Rojas, Elmir et al., 2016), neplata creditelor (Yeh and Lien, "
         "2009), telemarketing bancar (Moro, Cortez et al., 2014), Communities & Crime (Redmond and Baveja, "
         "2002), Mammographic mass (Elter, Schulz-Wendtland et al., 2007), Secondary mushroom (Wagner, Heider "
         "et al., 2021), Ailerons, două seturi KEEL (Alcalá-Fdez, Fernández et al., 2011) și două seturi de "
         "tranzacții de pe Kaggle; sursele sunt indicate sub Tabelul 2. Selecția designului și a "
         "hiperparametrilor a folosit o submulțime predefinită de 10 din cele 15 seturi de dezvoltare (Anexa "
         "A). Patru seturi de dezvoltare nu sunt dezechilibrate (32-50% minoritari); ele sunt păstrate ca "
         "seturi de control care arată dacă metoda se degradează la un dezechilibru slab, iar unul dintre ele, "
         "Mammographic mass, face parte din submulțimea de selecție. Cele 17 seturi de control nu au fost "
         "încărcate până la fixarea configurației: datele ULB de fraudă cu "
         "carduri, 284.807 tranzacții europene cu 492 de fraude (Dal Pozzolo, Caelen et al., 2015), rulate la "
         "dimensiune completă și prevalență nativă, și cele 16 seturi de referință derivate din KEEL și UCI "
         "compilate de Park and Im (2024) pentru Park, Lee et al. (2024), cu rapoarte de dezechilibru între "
         "10 și 42. Seturile de dezvoltare cu peste 50.000 de rânduri sunt subeșantionate stratificat, cu "
         "minimum 200 de cazuri pozitive, ceea ce ridică prevalența PaySim de la 0,13% la 0,40%."),
        ("p", "**Methods.** Ten baselines cover the main families: logistic regression with class weights; "
              "SMOTE followed by logistic regression, both with standardised features; random forest with "
              "balanced class weights (300 trees); Balanced Random Forest (300 trees); EasyEnsemble with "
              "AdaBoost members; RUSBoost (100 depth-3 trees); XGBoost and LightGBM with scale_pos_weight equal "
              "to the class ratio; unweighted LightGBM; and Self-paced Ensemble (50 trees). No method, including "
              "Cross-Balance, is tuned on the evaluation data, and all use library defaults or the settings "
              "stated above."
         if en else
         "**Metode.** Zece metode de referință acoperă principalele familii: regresie logistică cu ponderi de "
         "clasă și SMOTE urmat de regresie logistică (ambele cu variabile standardizate), pădure aleatoare cu "
         "ponderi echilibrate (300 de arbori), Balanced Random Forest (300 de arbori), EasyEnsemble cu membri "
         "AdaBoost, RUSBoost (100 de arbori de adâncime 3), XGBoost și LightGBM cu scale_pos_weight egal cu "
         "raportul claselor, LightGBM neponderat și Self-paced Ensemble (50 de arbori). Nicio metodă, inclusiv "
         "Cross-Balance, nu este calibrată pe datele de evaluare; toate folosesc valorile implicite ale "
         "bibliotecilor sau setările menționate."),
        ("p", "**Protocol.** Each dataset is evaluated with stratified five-fold cross-validation repeated "
              "three times (seeds 1 to 3), giving 15 test folds that are identical for all methods. Scaling and "
              "resampling are fitted inside training folds only, and every failed fit is logged. The primary "
              "measure is average precision (AP). ROC-AUC, R-precision (precision among the top n_{pos} "
              "scores), the Brier score and threshold measures at each method's own decision rule are "
              "reported as secondary measures. Methods are ranked per dataset by mean AP, with average ranks "
              "for ties. I test for differences with the Friedman test in the Iman-Davenport form (Iman and "
              "Davenport, 1980) and the Nemenyi critical difference, and compare Cross-Balance with each "
              "baseline using Wilcoxon signed-rank tests with Holm's (1979) correction (Demšar, 2006). As a "
              "separate check of decision rules, every method also receives an F1-optimal threshold estimated "
              "by the same internal three-fold cross-validation inside each training fold."
         if en else
         "**Protocol.** Fiecare set este evaluat prin validare încrucișată stratificată cu cinci partiții, "
         "repetată de trei ori (seed-urile 1-3), ceea ce dă 15 partiții de test identice pentru toate metodele. "
         "Scalarea și reeșantionarea se ajustează exclusiv pe partițiile de antrenare, iar orice eșec este "
         "înregistrat. Măsura principală este precizia medie (AP); ca măsuri secundare se raportează ROC-AUC, "
         "precizia R (precizia între primele n_{pos} scoruri), scorul Brier și indicatorii la pragul de decizie "
         "propriu fiecărei metode. Metodele sunt ordonate pe fiecare set după AP medie (egalitățile primesc "
         "rangul mediu). Diferențele sunt testate cu testul Friedman în forma Iman-Davenport (Iman and Davenport, "
         "1980) și diferența critică Nemenyi, iar Cross-Balance este comparat cu fiecare metodă de referință "
         "prin testul Wilcoxon cu corecția Holm (Holm, 1979; Demšar, 2006). Ca verificare separată a regulilor "
         "de decizie, fiecare metodă primește și un prag optim pentru F1, estimat prin aceeași validare "
         "încrucișată internă cu trei partiții în interiorul fiecărei partiții de antrenare."),
    ]

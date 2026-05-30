def get_law_data():
    # Input text and data
    years = []
    texts = []  
    laws = []
    topics = []
    chosen_laws = []

    #https://www.lavoro.gov.it/temi-e-priorita/previdenza/Pagine/Normativa
    years.append(2004)
    texts.append('Normativa per la pensioni, i contributi pensionistici, la previdenza e l\'inps.')
    laws.append(["2019|4","2015|208","2014|147",
                        "2013|147","2013|102","2012|228",
                        "2012|95","2011|201",
                        "2011|138","2011|98","2011|216",
                        "2011|183","2010|78",
                        "2009|78","2007|247",
                        "2006|42","2004|243"])
    topics.append("pensioni")
    chosen_laws.append(['2015|208', '2012|95', '2010|78'])

    # Anno 1997, Legge: 2008|145 -> Provo 2009|133
    years.append(1997)
    texts.append('Normativa per il trattamento delle sostanze chimiche (regolamento REACH).')
    laws.append(["2016|124","2014|27","2011|200",
                    "1997|281","2009|133","2008|145"])
    topics.append("sostanze")
    chosen_laws.append(['1997|281', '2014|27', '2011|200'])

    # Anno: 2012, Legge: 2015|150
    years.append(2012)
    texts.append('Normativa per l\'occupazione, il lavoro e i contratti di lavoro.')
    laws.append(["2014|183","2014|78","2013|99","2012|231",# conversioni in legge
                    "2014|34","2013|76","2012|207", "2013|102",
                    "2020|104","2020|149","2020|183","2021|41",
                    "2021|44", "2015|23","2015|150"])
    topics.append("lavoro")
    chosen_laws.append(['2015|150', '2020|183', '2021|44'])

    # Anno: 2005, Legge: 2017|51
    #https://www.mase.gov.it/pagina/combustibili-uso-trazione-normativa-nazionale
    #https://www.mase.gov.it/pagina/combustibili-uso-civile-industriale-e-marittimo-normativa-nazionale
    years.append(2005)
    texts.append("Normativa sui combustibili ad uso trazione, uso civile, industriale e marittimo.")
    laws.append(["2017|51","2005|66","2005|128","2006|152","2007|205","2014|112"])
    topics.append("combustibili")
    chosen_laws.append(['2005|66', '2005|128', '2017|51'])

    # DONE, CHOSEN 2010|31, per l'anno ho scelto 1997 (convenzione di Vienna, 1962 era troppo lontana)
    # https://www.mase.gov.it/pagina/normativa-di-riferimento-0
    years.append(1962)
    texts.append("Normativa sul nucleare e sulla gestione dei rifiuti radioattivi")
    laws.append(["2020|101","2014|45","2012|1",
                "2011|100","2010|31","2009|99",
                "2009|23","2005|282","2004|239",
                "2003|314","1970|1450","1962|1860"]) 
    topics.append("nucleare")
    chosen_laws.append(['2009|99', '2012|1', '2004|239'])

    # Anno: 1993, Legge: 1993|549
    #https://www.mase.gov.it/pagina/normativa
    years.append(1993)
    texts.append("Normativa, informazioni e obblighi per chi produce, utilizza, detiene le sostanze ozono lesive")
    laws.append(["1993|549","1996|56","2006|147",
                "1997|179","2002|179","2013|108",
                "2014|91","2001|35","2000|409"])
    topics.append("ozono")
    chosen_laws.append(['2014|91', '2002|179', '1993|549'])

    # DONE (pesticidi), ho scelto anno 1995, 2012|150
    # https://www.mase.gov.it/pagina/normativa-prodotti-fitosanitari
    years.append(1962)
    texts.append("Normativa prodotti fitosanitari per il controllo di qualsiasi organismo nocivo per le piante coltivate")
    laws.append(["2016|69","2012|55","2012|150",
                "1995|194","2001|290"])
    topics.append("pesticidi")
    chosen_laws.append(['1995|194', '2001|290', '2012|150'])

    # scelto 2014|86, anno 2010      
    years.append(2010)
    texts.append("Normativa riguardante i poteri speciali di Golden Power.")
    laws.append(["2020|179","2017|148","2022|187",
    "2012|21","2014|85","2020|180","2020|23",
    "2014|108","2019|22","2014|86","2014|35","2022|133"])
    chosen_laws.append(['2020|23', '2017|148', '2012|21'])
    topics.append("golden_power")

    return years, texts, laws, topics
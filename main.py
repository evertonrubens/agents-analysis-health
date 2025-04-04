from crewai import Agent, Task, Crew
from crewai_tools import SerperDevTool  # [Modificado] Importa ferramenta oficial de busca
# from crewai_tools import ScrapeWebsiteTool  # (Opcional: importar se for usar raspagem de sites)
from langchain_openai import ChatOpenAI      # Usando integração OpenAI com LangChain
from dotenv import load_dotenv
import os

# Carrega variáveis de ambiente
load_dotenv()
if not os.getenv("OPENAI_API_KEY"):
    raise EnvironmentError("A variável OPENAI_API_KEY não foi definida no arquivo .env.")

if not os.getenv("SERPER_API_KEY"):
    raise EnvironmentError("A variável SERPER_API_KEY não foi definida no arquivo .env.")


llm = ChatOpenAI(model="gpt-4-1106-preview", temperature=0.7)
search_tool = SerperDevTool(n_results=2)

#Agent Pesquisador
pesquisador = Agent(
    role="Pesquisador Científico",
    goal="Buscar os avanços mais recentes em IA aplicada à saúde. Trazer referências e links para os artigos.",
    backstory="Você é um cientista de dados experiente que trabalha com pesquisa aplicada em IA (Inteligência Artificial) para a área da Saúde. Seu papel é buscar fontes confiáveis e artigos recentes, permitindo ser do ano de 2024 até o momento.",
    tools=[search_tool],
    verbose=True,
    allow_delegation=True,
    llm=llm
)

#Agent Analista de dados
analista_dados = Agent(
    role="Analista de Dados Clínicos",
    goal="Interpretar dados clínicos e cruzar com as inovações em IA (Inteligência Artificial) para a área da Saúde.",
    backstory="Você é um cientista de dados com doutorado em bioinformática, especializado em cruzar dados clínicos com tecnologias emergentes.",
    llm=llm,
    verbose=True
)

#Agent Redador Técnico
redator_tecnico = Agent(
    role="Redator Técnico em IA (Inteligência Artificial) Médica",
    goal="Escrever um relatório técnico sobre inovações em IA (Inteligência Artificial) na saúde",
    backstory="Você é um redator técnico com experiência em publicações científicas e artigos de tecnologia médica.",
    llm=llm,
    verbose=True
)

#Agent Revisor Ético
revisor_etico = Agent(
    role="Revisor Ético",
    goal="Revisar o relatório para detectar vieses ou problemas éticos",
    backstory="Você é um especialista em bioética e conformidade em saúde digital. Seu papel é garantir neutralidade e responsabilidade.",
    llm=llm,
    verbose=True
)

#Agent Tradutor
tradutor = Agent(
    role="Tradutor Especializado em IA (Inteligência Artificial) Médica",
    goal="Traduzir o relatório para inglês ou português, mantendo a terminologia técnica",
    backstory="Você é um tradutor fluente em português e inglês com domínio do vocabulário técnico em IA (Inteligência Artificial) e medicina.",
    llm=llm,
    verbose=True
)


#Tarefas
task1 = Task(
    description="Pesquisar e resumir os 5 artigos mais recentes sobre IA na saúde.",
    expected_output="Lista com 5 artigos, fonte e resumo de cada um.",
    agent=pesquisador
)

task2 = Task(
    description="Interpretar os dados clínicos relacionados aos artigos e gerar uma análise cruzada.",
    expected_output="Correlações entre as inovações e aplicações clínicas.",
    agent=analista_dados
)

task3 = Task(
    description="Escrever um relatório técnico de 2 páginas com base na pesquisa e análise.",
    expected_output="Relatório técnico estruturado com introdução, desenvolvimento e conclusão.",
    agent=redator_tecnico
)

task4 = Task(
    description="Revisar o relatório identificando qualquer viés ou problema ético.",
    expected_output="Versão revisada do relatório com observações e correções.",
    agent=revisor_etico
)

task5 = Task(
    description="Traduzir o relatório final para português e inglês mantendo a terminologia técnica.",
    expected_output="Versões finais em ambos os idiomas.",
    agent=tradutor
)


crew = Crew(
    agents=[pesquisador, analista_dados, redator_tecnico, revisor_etico, tradutor],
    tasks=[task1, task2, task3, task4, task5],
    verbose=True
)

resultado_final = crew.kickoff() #Se você estiver executando em um servidor parrudo e com uma  conta business no OpenAI, use esse comando

#result1 = task1.execute_sync()                      # pesquisador e pobre igual a mim, execute estes comandos...
#result2 = task2.execute_sync(context=result1)       # pesquisador e pobre igual a mim, execute estes comandos...
#result3 = task3.execute_sync(context=result2)       # pesquisador e pobre igual a mim, execute estes comandos...
#result4 = task4.execute_sync(context=result3)       # pesquisador e pobre igual a mim, execute estes comandos...
#result5 = task5.execute_sync(context=result4)       # pesquisador e pobre igual a mim, execute estes comandos...
#resultado_final = result5
print("\n========== RELATÓRIO FINAL ==========\n")
print(resultado_final)
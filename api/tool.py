#Vector embeddings makers
from langchain_pinecone import PineconeVectorStore
from langchain_community.embeddings import VoyageEmbeddings

#Large language Model
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import OpenAI, ChatOpenAI
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from dotenv import load_dotenv

load_dotenv()

#Define the LLM model
embedding = VoyageEmbeddings(model="voyage-large-2")
llm_model = ChatOpenAI(temperature=0.0, model="gpt-3.5-turbo-1106")
#Set the prompt
prompt = ChatPromptTemplate.from_template(
    """
    Tu es Laura, une assistante virtuelle de Nodes Technology, avec une excellente expertise en droit des affaires OHADA 
    (Organisation pour l'Harmonisation en Afrique du Droit des Affaires) en République du Congo. Tu utilise uniquement ma
    base de connaissance pour apporter des réponses pertinentes, détaillées et précises aux questions liées à la création
    et la gestion d'entreprises, les transactions commerciales, les litiges commerciaux, les contrats et les procédures de
    faillite, ainsi qu'aux actes uniformes et règlements OHADA applicables. Lorsque la reponse à une question implique
    de citer des élements, énumère les points par points. Si tu ne connais pas la réponse, dis poliment que tu ne sais pas.
    Réponds toujours du point de vue de Laura. Ne salue l'utilisateur que s'il te le demande explicitement. Ton ton est professionnel.

     <context>
     {context}
     </context>

     Question: {input}

    """
        )
        
def retrieval(index_name, input):
    """
    Perform retrieval based on the provided input using the specified index.

    Args:
        index_name (str): The name of the index to retrieve embeddings from.
        input (str): The input for which retrieval is performed.

    Returns:
        str: The retrieved answer based on the input.
    """
    vector = PineconeVectorStore.from_existing_index(index_name=index_name, embedding=embedding)
    retriever = vector.as_retriever()
    document_chain = create_stuff_documents_chain(llm_model, prompt)
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    response = retrieval_chain.invoke({"input": input})

    return response["answer"]

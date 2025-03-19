from dotenv import load_dotenv
import os

from langchain_community.graphs import Neo4jGraph

# Warning control
import warnings
warnings.filterwarnings("ignore")


# Load all the required variables for Neo4J connection
load_dotenv('.env', override=True)
NEO4J_URI = os.getenv('NEO4J_URI')
NEO4J_USERNAME = os.getenv('NEO4J_USERNAME')
NEO4J_PASSWORD = os.getenv('NEO4J_PASSWORD')
NEO4J_DATABASE = os.getenv('NEO4J_DATABASE')

# Create the connection
kg = Neo4jGraph(
    url=NEO4J_URI, username=NEO4J_USERNAME, password=NEO4J_PASSWORD, database=NEO4J_DATABASE
)

#Match any
cypher = """
  MATCH (n) 
  RETURN count(n)
  """

# Return of specific node type and with specific properties (here the title) and from there only return the release and tagline
cypher2 = """
  MATCH (cloudAtlas:Movie {title:"Cloud Atlas"}) 
  RETURN cloudAtlas.released, cloudAtlas.tagline
  """

# We can have more complex conditions in the WHERE clause
cypher3 = """
  MATCH (nineties:Movie) 
  WHERE nineties.released >= 1990 
    AND nineties.released < 2000 
  RETURN nineties.title
  """

# We can define relationship matching
cypher4 = """
  MATCH (actor:Person)-[:ACTED_IN]->(movie:Movie) 
  RETURN actor.name, movie.title LIMIT 10
  """
# We can filter
cypher5 = """
  MATCH (tom:Person {name: "Tom Hanks"})-[:ACTED_IN]->(tomHanksMovies:Movie) 
  RETURN tom.name,tomHanksMovies.title
  """

# We can delete parts of the graph
cypher6 = """
    MATCH (emil:Person {name:"Emil Eifrem"})-[actedIn:ACTED_IN]->(movie:Movie)
    RETURN emil.name, movie.title
    """
# We can add new nodes to the graph
cypher7 = """
    CREATE (andreas:Person {name:"Andreas"})
    RETURN andreas
    """
# And even entire relationships
cypher = """
    MATCH (andreas:Person {name:"Andreas"}), (emil:Person {name:"Emil Eifrem"})
    MERGE (andreas)-[hasRelationship:WORKS_WITH]->(emil)
    RETURN andreas, hasRelationship, emil
    """

#Run the query
result = kg.query(cypher)
print(result)
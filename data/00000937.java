public class GraphSpecTest { @ Test public void serialization ( ) throws IOException { JSONGraphSpec graphSpec = new JSONGraphSpec ( "Span" ) ; graphSpec . setValuesPointer ( "/0" ) . addProperty ( "trace_id" , "/trace_id" , null ) . addProperty ( "id" , "/id" , null ) ; graphSpec . addEdge ( "PARENT" ) . addTargetLabel ( "Span" ) . addTargetProperty ( "id" , "/parent_id" , null ) ; ObjectMapper mapper = new ObjectMapper ( ) ; String initJson = mapper . writerWithDefaultPrettyPrinter ( ) . writeValueAsString ( graphSpec ) ; JSONGraphSpec result = mapper . readerFor ( JSONGraphSpec . class ) . readValue ( initJson ) ; String resultJson = mapper . writerWithDefaultPrettyPrinter ( ) . writeValueAsString ( result ) ; assertEquals ( initJson , resultJson ) ; } }
public class TikaTest { private Document doc ; @ Before public void setup ( ) { doc = new Document ( ) ; } @ Test public void testPDF ( ) throws IOException { parse ( "paxos-simple . pdf" , "application/pdf" , "foo" ) ; assertThat ( doc . getField ( "foo" ) , not ( nullValue ( ) ) ) ; } @ Test public void testXML ( ) throws IOException { parse ( "example . xml" , "text/xml" , "bar" ) ; assertThat ( doc . getField ( "bar" ) , not ( nullValue ( ) ) ) ; } @ Test public void testWord ( ) throws IOException { parse ( "example . doc" , "application/msword" , "bar" ) ; assertThat ( doc . getField ( "bar" ) , not ( nullValue ( ) ) ) ; assertThat ( doc . get ( "bar" ) , containsString ( "The express mission of the organization" ) ) ; } private void parse ( final String resource , final String type , final String field ) throws IOException { final InputStream in = getClass ( ) . getClassLoader ( ) . getResourceAsStream ( resource ) ; try { Tika . INSTANCE . parse ( in , type , field , doc ) ; } finally { in . close ( ) ; } } }
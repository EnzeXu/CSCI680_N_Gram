public class DesignDocumentTest { @ Test ( expected = IllegalArgumentException . class ) public void notDesignDocument ( ) throws Exception { new DesignDocument ( new JSONObject ( " { _id : \"hello\" } " ) ) ; } @ Test public void noViews ( ) throws Exception { final DesignDocument ddoc = new DesignDocument ( new JSONObject ( " { _id : \"_design/hello\" } " ) ) ; assertThat ( ddoc . getAllViews ( ) . size ( ) , is ( 0 ) ) ; } @ Test public void views ( ) throws Exception { final JSONObject view = new JSONObject ( ) ; view . put ( "index" , "function ( doc ) { return null ; } " ) ; final JSONObject fulltext = new JSONObject ( ) ; fulltext . put ( "foo" , view ) ; final JSONObject json = new JSONObject ( ) ; json . put ( "_id" , "_design/hello" ) ; json . put ( "fulltext" , fulltext ) ; final DesignDocument ddoc = new DesignDocument ( json ) ; assertThat ( ddoc . getView ( "foo" ) , notNullValue ( ) ) ; assertThat ( ddoc . getAllViews ( ) . size ( ) , is ( 1 ) ) ; } }
public class PwGroupTest { PwGroupV3 mPG ; @ Before public void setUp ( ) throws Exception { Context ctx = InstrumentationRegistry . getInstrumentation ( ) . getTargetContext ( ) ; mPG = ( PwGroupV3 ) TestData . GetTest1 ( ctx ) . getGroups ( ) . get ( 0 ) ; } @ Test public void testGroupName ( ) { assertTrue ( "Name was " + mPG . name , mPG . name . equals ( "Internet" ) ) ; } }
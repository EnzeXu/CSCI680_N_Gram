public class ScriptFilter extends ScriptOperation implements Filter < ScriptOperation . Context > { @ ConstructorProperties ( { "script" } ) public ScriptFilter ( String script ) { super ( ANY , script , Boolean . class ) ; } @ ConstructorProperties ( { "script" , "parameterName" , "parameterType" } ) public ScriptFilter ( String script , String parameterName , Class parameterType ) { super ( 1 , script , Boolean . class , new String [ ] { parameterName } , new Class [ ] { parameterType } ) ; } @ ConstructorProperties ( { "script" , "expectedTypes" } ) public ScriptFilter ( String script , Class [ ] expectedTypes ) { super ( expectedTypes . length , script , Boolean . class , expectedTypes ) ; } @ ConstructorProperties ( { "script" , "parameterNames" , "parameterTypes" } ) public ScriptFilter ( String script , String [ ] parameterNames , Class [ ] parameterTypes ) { super ( parameterTypes . length , script , Boolean . class , parameterNames , parameterTypes ) ; } public String getScript ( ) { return getBlock ( ) ; } @ Override public boolean isRemove ( FlowProcess flowProcess , FilterCall < Context > filterCall ) { return ( Boolean ) evaluate ( filterCall . getContext ( ) , filterCall . getArguments ( ) ) ; } }
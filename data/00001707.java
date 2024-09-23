public class ScriptTupleFunction extends ScriptOperation implements Function<ScriptOperation.Context> { @ConstructorProperties({"fieldDeclaration", "script"}) public ScriptTupleFunction( Fields fieldDeclaration, String script ) { super( ANY, fieldDeclaration, script, Tuple.class ); } @ConstructorProperties({"fieldDeclaration", "script", "expectedTypes"}) public ScriptTupleFunction( Fields fieldDeclaration, String script, Class[] expectedTypes ) { super( expectedTypes.length, fieldDeclaration, script, Tuple.class, expectedTypes ); } @ConstructorProperties({"fieldDeclaration", "script", "parameterNames", "parameterTypes"}) public ScriptTupleFunction( Fields fieldDeclaration, String script, String[] parameterNames, Class[] parameterTypes ) { super( parameterTypes.length, fieldDeclaration, script, Tuple.class, parameterNames, parameterTypes ); } public String getScript() { return getBlock(); } @Override public void operate( FlowProcess flowProcess, FunctionCall<Context> functionCall ) { functionCall.getOutputCollector().add( (Tuple) evaluate( functionCall.getContext(), functionCall.getArguments() ) ); } }
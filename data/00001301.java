public class Debug extends BaseOperation<Long> implements Filter<Long>, PlannedOperation<Long> { static public enum Output { STDOUT, STDERR } private Output output = Output.STDERR; private String prefix = null; private boolean printFields = false; private int printFieldsEvery = 10; private int printTupleEvery = 1; public Debug() { } @ConstructorProperties({"prefix"}) public Debug( String prefix ) { this.prefix = prefix; } @ConstructorProperties({"prefix", "printFields"}) public Debug( String prefix, boolean printFields ) { this.prefix = prefix; this.printFields = printFields; } @ConstructorProperties({"printFields"}) public Debug( boolean printFields ) { this.printFields = printFields; } @ConstructorProperties({"output"}) public Debug( Output output ) { this.output = output; } @ConstructorProperties({"output", "prefix"}) public Debug( Output output, String prefix ) { this.output = output; this.prefix = prefix; } @ConstructorProperties({"output", "prefix", "printFields"}) public Debug( Output output, String prefix, boolean printFields ) { this.output = output; this.prefix = prefix; this.printFields = printFields; } @ConstructorProperties({"output", "printFields"}) public Debug( Output output, boolean printFields ) { this.output = output; this.printFields = printFields; } public Output getOutput() { return output; } public String getPrefix() { return prefix; } public boolean isPrintFields() { return printFields; } public int getPrintFieldsEvery() { return printFieldsEvery; } public void setPrintFieldsEvery( int printFieldsEvery ) { this.printFieldsEvery = printFieldsEvery; } public int getPrintTupleEvery() { return printTupleEvery; } public void setPrintTupleEvery( int printTupleEvery ) { this.printTupleEvery = printTupleEvery; } @Override public boolean supportsPlannerLevel( PlannerLevel plannerLevel ) { return plannerLevel instanceof DebugLevel; } @Override public void prepare( FlowProcess flowProcess, OperationCall<Long> operationCall ) { super.prepare( flowProcess, operationCall ); operationCall.setContext( 0L ); } public boolean isRemove( FlowProcess flowProcess, FilterCall<Long> filterCall ) { PrintStream stream = output == Output.STDOUT ? System.out : System.err; if( printFields && filterCall.getContext() % printFieldsEvery == 0 ) print( stream, filterCall.getArguments().getFields().print() ); if( filterCall.getContext() % printTupleEvery == 0 ) print( stream, filterCall.getArguments().getTuple().print() ); filterCall.setContext( filterCall.getContext() + 1 ); return false; } @Override public void cleanup( FlowProcess flowProcess, OperationCall<Long> longOperationCall ) { if( longOperationCall.getContext() == null ) return; PrintStream stream = output == Output.STDOUT ? System.out : System.err; print( stream, "tuples count: " + longOperationCall.getContext().toString() ); } private void print( PrintStream stream, String message ) { if( prefix != null ) { stream.print( prefix ); stream.print( ": " ); } stream.println( message ); } @Override public boolean equals( Object object ) { if( this == object ) return true; if( !( object instanceof Debug ) ) return false; if( !super.equals( object ) ) return false; Debug debug = (Debug) object; if( printFields != debug.printFields ) return false; if( printFieldsEvery != debug.printFieldsEvery ) return false; if( printTupleEvery != debug.printTupleEvery ) return false; if( output != debug.output ) return false; if( prefix != null ? !prefix.equals( debug.prefix ) : debug.prefix != null ) return false; return true; } @Override public int hashCode() { int result = super.hashCode(); result = 31 * result + ( output != null ? output.hashCode() : 0 ); result = 31 * result + ( prefix != null ? prefix.hashCode() : 0 ); result = 31 * result + ( printFields ? 1 : 0 ); result = 31 * result + printFieldsEvery; result = 31 * result + printTupleEvery; return result; } }
public class Limit extends BaseOperation<Limit.Context> implements Filter<Limit.Context> { private long limit = 0; public static class Context { public long limit = 0; public long count = 0; public boolean increment() { if( limit == count ) return true; count++; return false; } } @ConstructorProperties({"limit"}) public Limit( long limit ) { this.limit = limit; } @Property(name = "limit", visibility = Visibility.PUBLIC) @PropertyDescription("The upper limit.") public long getLimit() { return limit; } @Override public void prepare( FlowProcess flowProcess, OperationCall<Context> operationCall ) { super.prepare( flowProcess, operationCall ); Context context = new Context(); operationCall.setContext( context ); int numTasks = flowProcess.getNumProcessSlices(); int taskNum = flowProcess.getCurrentSliceNum(); context.limit = (long) Math.floor( (double) limit / (double) numTasks ); long remainingLimit = limit % numTasks; context.limit += taskNum < remainingLimit ? 1 : 0; } @Override public boolean isRemove( FlowProcess flowProcess, FilterCall<Context> filterCall ) { return filterCall.getContext().increment(); } @Override public boolean equals( Object object ) { if( this == object ) return true; if( !( object instanceof Limit ) ) return false; if( !super.equals( object ) ) return false; Limit limit1 = (Limit) object; if( limit != limit1.limit ) return false; return true; } @Override public int hashCode() { int result = super.hashCode(); result = 31 * result + (int) ( limit ^ limit >>> 32 ); return result; } }
public class FilterNotNull extends BaseOperation implements Filter { @Override public boolean isRemove( FlowProcess flowProcess, FilterCall filterCall ) { for( Object value : filterCall.getArguments().getTuple() ) { if( value != null ) return true; } return false; } }
public class And extends Logic { @ConstructorProperties({"filters"}) public And( Filter... filters ) { super( filters ); } @ConstructorProperties({"lhsArgumentsSelector", "lhsFilter", "rhsArgumentSelector", "rhsFilter"}) public And( Fields lhsArgumentSelector, Filter lhsFilter, Fields rhsArgumentSelector, Filter rhsFilter ) { super( lhsArgumentSelector, lhsFilter, rhsArgumentSelector, rhsFilter ); } @ConstructorProperties({"argumentFilters", "filters"}) public And( Fields[] argumentSelectors, Filter[] filters ) { super( argumentSelectors, filters ); } @Override public boolean isRemove( FlowProcess flowProcess, FilterCall filterCall ) { TupleEntry arguments = filterCall.getArguments(); Context context = (Context) filterCall.getContext(); TupleEntry[] argumentEntries = context.argumentEntries; for( int i = 0; i < argumentSelectors.length; i++ ) { Tuple selected = arguments.selectTuple( argumentSelectors[ i ] ); argumentEntries[ i ].setTuple( selected ); if( !filters[ i ].isRemove( flowProcess, context.calls[ i ] ) ) return false; } return true; } }
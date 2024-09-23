public class Xor extends Logic { @ConstructorProperties({"filters"}) public Xor( Filter... filters ) { super( filters ); } @ConstructorProperties({"lhsArgumentsSelector", "lhsFilter", "rhsArgumentSelector", "rhsFilter"}) public Xor( Fields lhsArgumentSelector, Filter lhsFilter, Fields rhsArgumentSelector, Filter rhsFilter ) { super( lhsArgumentSelector, lhsFilter, rhsArgumentSelector, rhsFilter ); } @Override public boolean isRemove( FlowProcess flowProcess, FilterCall filterCall ) { Context context = (Logic.Context) filterCall.getContext(); TupleEntry lhsEntry = context.argumentEntries[ 0 ]; TupleEntry rhsEntry = context.argumentEntries[ 1 ]; lhsEntry.setTuple( filterCall.getArguments().selectTuple( argumentSelectors[ 0 ] ) ); rhsEntry.setTuple( filterCall.getArguments().selectTuple( argumentSelectors[ 1 ] ) ); boolean lhsResult = filters[ 0 ].isRemove( flowProcess, context.calls[ 0 ] ); boolean rhsResult = filters[ 1 ].isRemove( flowProcess, context.calls[ 1 ] ); return lhsResult != rhsResult; } }
public class Splice extends Pipe { enum Kind { GroupBy, CoGroup, Merge, Join } private Kind kind; private String spliceName; private final List<Pipe> pipes = new ArrayList<Pipe>(); protected final Map<String, Fields> keyFieldsMap = new LinkedHashMap<String, Fields>(); protected Map<String, Fields> sortFieldsMap = new LinkedHashMap<String, Fields>(); private boolean reverseOrder = false; protected Fields declaredFields; protected Fields resultGroupFields; private int numSelfJoins = 0; private Joiner joiner; private transient Map<String, Integer> pipePos; protected Splice( Pipe lhs, Fields lhsGroupFields, Pipe rhs, Fields rhsGroupFields, Fields declaredFields ) { this( lhs, lhsGroupFields, rhs, rhsGroupFields, declaredFields, null, null ); } protected Splice( Pipe lhs, Fields lhsGroupFields, Pipe rhs, Fields rhsGroupFields, Fields declaredFields, Fields resultGroupFields ) { this( lhs, lhsGroupFields, rhs, rhsGroupFields, declaredFields, resultGroupFields, null ); } protected Splice( Pipe lhs, Fields lhsGroupFields, Pipe rhs, Fields rhsGroupFields, Fields declaredFields, Joiner joiner ) { this( Pipe.pipes( lhs, rhs ), Fields.fields( lhsGroupFields, rhsGroupFields ), declaredFields, joiner ); } protected Splice( Pipe lhs, Fields lhsGroupFields, Pipe rhs, Fields rhsGroupFields, Fields declaredFields, Fields resultGroupFields, Joiner joiner ) { this( Pipe.pipes( lhs, rhs ), Fields.fields( lhsGroupFields, rhsGroupFields ), declaredFields, resultGroupFields, joiner ); } protected Splice( Pipe lhs, Fields lhsGroupFields, Pipe rhs, Fields rhsGroupFields, Joiner joiner ) { this( lhs, lhsGroupFields, rhs, rhsGroupFields, null, joiner ); } protected Splice( Pipe lhs, Fields lhsGroupFields, Pipe rhs, Fields rhsGroupFields ) { this( Pipe.pipes( lhs, rhs ), Fields.fields( lhsGroupFields, rhsGroupFields ) ); } protected Splice( Pipe... pipes ) { this( pipes, (Fields[]) null ); } protected Splice( Pipe[] pipes, Fields[] groupFields ) { this( null, pipes, groupFields, null, null ); } protected Splice( String spliceName, Pipe[] pipes, Fields[] groupFields ) { this( spliceName, pipes, groupFields, null, null ); } protected Splice( String spliceName, Pipe[] pipes, Fields[] groupFields, Fields declaredFields ) { this( spliceName, pipes, groupFields, declaredFields, null ); } protected Splice( String spliceName, Pipe[] pipes, Fields[] groupFields, Fields declaredFields, Fields resultGroupFields ) { this( spliceName, pipes, groupFields, declaredFields, resultGroupFields, null ); } protected Splice( Pipe[] pipes, Fields[] groupFields, Fields declaredFields, Joiner joiner ) { this( null, pipes, groupFields, declaredFields, null, joiner ); } protected Splice( Pipe[] pipes, Fields[] groupFields, Fields declaredFields, Fields resultGroupFields, Joiner joiner ) { this( null, pipes, groupFields, declaredFields, resultGroupFields, joiner ); } protected Splice( String spliceName, Pipe[] pipes, Fields[] groupFields, Fields declaredFields, Fields resultGroupFields, Joiner joiner ) { if( pipes == null ) throw new IllegalArgumentException( "pipes array may not be null" ); setKind(); this.spliceName = spliceName; int uniques = new HashSet<Pipe>( asList( Pipe.resolvePreviousAll( pipes ) ) ).size(); if( pipes.length > 1 && uniques == 1 ) { if( isMerge() ) throw new IllegalArgumentException( "may not merge a pipe with itself without intermediate operations after the split" ); if( groupFields == null ) throw new IllegalArgumentException( "groupFields array may not be null" ); if( new HashSet<Fields>( asList( groupFields ) ).size() != 1 ) throw new IllegalArgumentException( "all groupFields must be identical" ); addPipe( pipes[ 0 ] ); this.numSelfJoins = pipes.length - 1; this.keyFieldsMap.put( pipes[ 0 ].getName(), groupFields[ 0 ] ); if( resultGroupFields != null && groupFields[ 0 ].size() * pipes.length != resultGroupFields.size() ) throw new IllegalArgumentException( "resultGroupFields and cogroup joined fields must be same size" ); } else { int last = -1; for( int i = 0; i < pipes.length; i++ ) { addPipe( pipes[ i ] ); if( groupFields == null || groupFields.length == 0 ) { addGroupFields( pipes[ i ], Fields.FIRST ); continue; } if( last != -1 && last != groupFields[ i ].size() ) throw new IllegalArgumentException( "all groupFields must be same size" ); last = groupFields[ i ].size(); addGroupFields( pipes[ i ], groupFields[ i ] ); } if( resultGroupFields != null && last * pipes.length != resultGroupFields.size() ) throw new IllegalArgumentException( "resultGroupFields and cogroup resulting joined fields must be same size" ); } this.declaredFields = declaredFields; this.resultGroupFields = resultGroupFields; this.joiner = joiner; verifyCoGrouper(); } protected Splice( String spliceName, Pipe lhs, Fields lhsGroupFields, Pipe rhs, Fields rhsGroupFields, Fields declaredFields ) { this( lhs, lhsGroupFields, rhs, rhsGroupFields, declaredFields ); this.spliceName = spliceName; } protected Splice( String spliceName, Pipe lhs, Fields lhsGroupFields, Pipe rhs, Fields rhsGroupFields, Fields declaredFields, Fields resultGroupFields ) { this( lhs, lhsGroupFields, rhs, rhsGroupFields, declaredFields, resultGroupFields ); this.spliceName = spliceName; } protected Splice( String spliceName, Pipe lhs, Fields lhsGroupFields, Pipe rhs, Fields rhsGroupFields, Fields declaredFields, Joiner joiner ) { this( lhs, lhsGroupFields, rhs, rhsGroupFields, declaredFields, joiner ); this.spliceName = spliceName; } protected Splice( String spliceName, Pipe lhs, Fields lhsGroupFields, Pipe rhs, Fields rhsGroupFields, Fields declaredFields, Fields resultGroupFields, Joiner joiner ) { this( lhs, lhsGroupFields, rhs, rhsGroupFields, declaredFields, resultGroupFields, joiner ); this.spliceName = spliceName; } protected Splice( String spliceName, Pipe lhs, Fields lhsGroupFields, Pipe rhs, Fields rhsGroupFields, Joiner joiner ) { this( lhs, lhsGroupFields, rhs, rhsGroupFields, joiner ); this.spliceName = spliceName; } protected Splice( String spliceName, Pipe lhs, Fields lhsGroupFields, Pipe rhs, Fields rhsGroupFields ) { this( lhs, lhsGroupFields, rhs, rhsGroupFields ); this.spliceName = spliceName; } protected Splice( String spliceName, Pipe... pipes ) { this( pipes ); this.spliceName = spliceName; } protected Splice( Pipe pipe, Fields groupFields, int numSelfJoins, Fields declaredFields ) { this( pipe, groupFields, numSelfJoins ); this.declaredFields = declaredFields; } protected Splice( Pipe pipe, Fields groupFields, int numSelfJoins, Fields declaredFields, Fields resultGroupFields ) { this( pipe, groupFields, numSelfJoins ); this.declaredFields = declaredFields; this.resultGroupFields = resultGroupFields; if( resultGroupFields != null && groupFields.size() * numSelfJoins != resultGroupFields.size() ) throw new IllegalArgumentException( "resultGroupFields and cogroup resulting join fields must be same size" ); } protected Splice( Pipe pipe, Fields groupFields, int numSelfJoins, Fields declaredFields, Joiner joiner ) { this( pipe, groupFields, numSelfJoins, declaredFields ); this.joiner = joiner; verifyCoGrouper(); } protected Splice( Pipe pipe, Fields groupFields, int numSelfJoins, Fields declaredFields, Fields resultGroupFields, Joiner joiner ) { this( pipe, groupFields, numSelfJoins, declaredFields, resultGroupFields ); this.joiner = joiner; verifyCoGrouper(); } protected Splice( Pipe pipe, Fields groupFields, int numSelfJoins, Joiner joiner ) { setKind(); addPipe( pipe ); this.keyFieldsMap.put( pipe.getName(), groupFields ); this.numSelfJoins = numSelfJoins; this.joiner = joiner; verifyCoGrouper(); } protected Splice( Pipe pipe, Fields groupFields, int numSelfJoins ) { this( pipe, groupFields, numSelfJoins, (Joiner) null ); } protected Splice( String spliceName, Pipe pipe, Fields groupFields, int numSelfJoins, Fields declaredFields ) { this( pipe, groupFields, numSelfJoins, declaredFields ); this.spliceName = spliceName; } protected Splice( String spliceName, Pipe pipe, Fields groupFields, int numSelfJoins, Fields declaredFields, Fields resultGroupFields ) { this( pipe, groupFields, numSelfJoins, declaredFields, resultGroupFields ); this.spliceName = spliceName; } protected Splice( String spliceName, Pipe pipe, Fields groupFields, int numSelfJoins, Fields declaredFields, Joiner joiner ) { this( pipe, groupFields, numSelfJoins, declaredFields, joiner ); this.spliceName = spliceName; } protected Splice( String spliceName, Pipe pipe, Fields groupFields, int numSelfJoins, Fields declaredFields, Fields resultGroupFields, Joiner joiner ) { this( pipe, groupFields, numSelfJoins, declaredFields, resultGroupFields, joiner ); this.spliceName = spliceName; } protected Splice( String spliceName, Pipe pipe, Fields groupFields, int numSelfJoins, Joiner joiner ) { this( pipe, groupFields, numSelfJoins, joiner ); this.spliceName = spliceName; } protected Splice( String spliceName, Pipe pipe, Fields groupFields, int numSelfJoins ) { this( pipe, groupFields, numSelfJoins ); this.spliceName = spliceName; } protected Splice( Pipe pipe ) { this( null, pipe, Fields.ALL, null, false ); } protected Splice( Pipe pipe, Fields groupFields ) { this( null, pipe, groupFields, null, false ); } protected Splice( String spliceName, Pipe pipe, Fields groupFields ) { this( spliceName, pipe, groupFields, null, false ); } protected Splice( Pipe pipe, Fields groupFields, Fields sortFields ) { this( null, pipe, groupFields, sortFields, false ); } protected Splice( String spliceName, Pipe pipe, Fields groupFields, Fields sortFields ) { this( spliceName, pipe, groupFields, sortFields, false ); } protected Splice( Pipe pipe, Fields groupFields, Fields sortFields, boolean reverseOrder ) { this( null, pipe, groupFields, sortFields, reverseOrder ); } protected Splice( String spliceName, Pipe pipe, Fields groupFields, Fields sortFields, boolean reverseOrder ) { this( spliceName, Pipe.pipes( pipe ), groupFields, sortFields, reverseOrder ); } protected Splice( Pipe[] pipes, Fields groupFields ) { this( null, pipes, groupFields, null, false ); } protected Splice( String spliceName, Pipe[] pipes, Fields groupFields ) { this( spliceName, pipes, groupFields, null, false ); } protected Splice( Pipe[] pipes, Fields groupFields, Fields sortFields ) { this( null, pipes, groupFields, sortFields, false ); } protected Splice( String spliceName, Pipe[] pipe, Fields groupFields, Fields sortFields ) { this( spliceName, pipe, groupFields, sortFields, false ); } protected Splice( Pipe[] pipes, Fields groupFields, Fields sortFields, boolean reverseOrder ) { this( null, pipes, groupFields, sortFields, reverseOrder ); } protected Splice( String spliceName, Pipe[] pipes, Fields groupFields, Fields sortFields, boolean reverseOrder ) { if( pipes == null ) throw new IllegalArgumentException( "pipes array may not be null" ); if( groupFields == null ) throw new IllegalArgumentException( "groupFields may not be null" ); setKind(); this.spliceName = spliceName; for( Pipe pipe : pipes ) { addPipe( pipe ); this.keyFieldsMap.put( pipe.getName(), groupFields ); if( sortFields != null ) this.sortFieldsMap.put( pipe.getName(), sortFields ); } this.reverseOrder = reverseOrder; this.joiner = new InnerJoin(); } private void verifyCoGrouper() { if( isJoin() && joiner instanceof BufferJoin ) throw new IllegalArgumentException( "invalid joiner, may not use BufferJoiner in a HashJoin" ); if( joiner == null ) { joiner = new InnerJoin(); return; } if( joiner.numJoins() == -1 ) return; int joins = Math.max( numSelfJoins, keyFieldsMap.size() - 1 ); if( joins != joiner.numJoins() ) throw new IllegalArgumentException( "invalid joiner, only accepts " + joiner.numJoins() + " joins, there are: " + joins ); } private void setKind() { if( this instanceof GroupBy ) kind = Kind.GroupBy; else if( this instanceof CoGroup ) kind = Kind.CoGroup; else if( this instanceof Merge ) kind = Kind.Merge; else kind = Kind.Join; } public Fields getDeclaredFields() { return declaredFields; } private void addPipe( Pipe pipe ) { if( pipe.getName() == null ) throw new IllegalArgumentException( "each input pipe must have a name" ); pipes.add( pipe ); } private void addGroupFields( Pipe pipe, Fields fields ) { if( keyFieldsMap.containsKey( pipe.getName() ) ) throw new IllegalArgumentException( "each input pipe branch must be uniquely named" ); keyFieldsMap.put( pipe.getName(), fields ); } @Override public String getName() { if( spliceName != null ) return spliceName; StringBuffer buffer = new StringBuffer(); for( Pipe pipe : pipes ) { if( buffer.length() != 0 ) { if( isGroupBy() || isMerge() ) buffer.append( "+" ); else if( isCoGroup() || isJoin() ) buffer.append( "*" ); } buffer.append( pipe.getName() ); } spliceName = buffer.toString(); return spliceName; } @Override public Pipe[] getPrevious() { return pipes.toArray( new Pipe[ pipes.size() ] ); } public Map<String, Fields> getKeySelectors() { return keyFieldsMap; } public Map<String, Fields> getSortingSelectors() { return sortFieldsMap; } public boolean isSorted() { return !sortFieldsMap.isEmpty(); } public boolean isSortReversed() { return reverseOrder; } public synchronized Map<String, Integer> getPipePos() { if( pipePos != null ) return pipePos; pipePos = new HashMap<String, Integer>(); int pos = 0; for( Object pipe : pipes ) pipePos.put( ( (Pipe) pipe ).getName(), pos++ ); return pipePos; } public Joiner getJoiner() { return joiner; } public final boolean isGroupBy() { return kind == Kind.GroupBy; } public final boolean isCoGroup() { return kind == Kind.CoGroup; } public final boolean isMerge() { return kind == Kind.Merge; } public final boolean isJoin() { return kind == Kind.Join; } public int getNumSelfJoins() { return numSelfJoins; } public boolean isSelfJoin() { return numSelfJoins != 0; } @Override public Scope outgoingScopeFor( Set<Scope> incomingScopes ) { Map<String, Fields> groupingSelectors = resolveGroupingSelectors( incomingScopes ); Map<String, Fields> sortingSelectors = resolveSortingSelectors( incomingScopes ); Fields declared = resolveDeclared( incomingScopes ); Fields outGroupingFields = resultGroupFields; if( outGroupingFields == null && isCoGroup() ) outGroupingFields = createJoinFields( incomingScopes, groupingSelectors, declared ); Scope.Kind kind = getScopeKind(); return new Scope( getName(), declared, outGroupingFields, groupingSelectors, sortingSelectors, declared, kind ); } private Scope.Kind getScopeKind() { switch( kind ) { case GroupBy: return Scope.Kind.GROUPBY; case CoGroup: return Scope.Kind.COGROUP; case Merge: return Scope.Kind.MERGE; case Join: return Scope.Kind.HASHJOIN; } throw new IllegalStateException( "unknown kind: " + kind ); } private Fields createJoinFields( Set<Scope> incomingScopes, Map<String, Fields> groupingSelectors, Fields declared ) { if( declared.isNone() ) declared = Fields.UNKNOWN; Map<String, Fields> incomingFields = new HashMap<String, Fields>(); for( Scope scope : incomingScopes ) incomingFields.put( scope.getName(), scope.getIncomingSpliceFields() ); Fields outGroupingFields = Fields.NONE; int offset = 0; for( Pipe pipe : pipes ) { String pipeName = pipe.getName(); Fields pipeGroupingSelector = groupingSelectors.get( pipeName ); Fields incomingField = incomingFields.get( pipeName ); if( !pipeGroupingSelector.isNone() ) { Fields offsetFields = incomingField.selectPos( pipeGroupingSelector, offset ); Fields resolvedSelect = declared.select( offsetFields ); outGroupingFields = outGroupingFields.append( resolvedSelect ); } offset += incomingField.size(); } return outGroupingFields; } Map<String, Fields> resolveGroupingSelectors( Set<Scope> incomingScopes ) { try { Map<String, Fields> groupingSelectors = getKeySelectors(); Map<String, Fields> groupingFields = resolveSelectorsAgainstIncoming( incomingScopes, groupingSelectors, "grouping" ); if( !verifySameSize( groupingFields ) ) throw new OperatorException( this, "all grouping fields must be same size: " + toString() ); verifySameTypes( groupingSelectors, groupingFields ); return groupingFields; } catch( FieldsResolverException exception ) { throw new OperatorException( this, OperatorException.Kind.grouping, exception.getSourceFields(), exception.getSelectorFields(), exception ); } catch( RuntimeException exception ) { throw new OperatorException( this, "could not resolve grouping selector in: " + this, exception ); } } private boolean verifySameTypes( Map<String, Fields> groupingSelectors, Map<String, Fields> groupingFields ) { boolean[] hasComparator = new boolean[ groupingFields.values().iterator().next().size() ]; for( Map.Entry<String, Fields> entry : groupingSelectors.entrySet() ) { Comparator[] comparatorsArray = entry.getValue().getComparators(); for( int i = 0; i < comparatorsArray.length; i++ ) hasComparator[ i ] = hasComparator[ i ] || comparatorsArray[ i ] != null; } Iterator<Fields> iterator = groupingFields.values().iterator(); Fields lhsFields = iterator.next(); Type[] lhsTypes = lhsFields.getTypes(); if( lhsTypes == null ) return true; while( iterator.hasNext() ) { Fields rhsFields = iterator.next(); Type[] rhsTypes = rhsFields.getTypes(); if( rhsTypes == null ) return true; for( int i = 0; i < lhsTypes.length; i++ ) { if( hasComparator[ i ] ) continue; Type lhs = lhsTypes[ i ]; Type rhs = rhsTypes[ i ]; lhs = getCanonicalType( lhs ); rhs = getCanonicalType( rhs ); if( lhs.equals( rhs ) ) continue; Fields lhsError = new Fields( lhsFields.get( i ), lhsFields.getType( i ) ); Fields rhsError = new Fields( rhsFields.get( i ), rhsFields.getType( i ) ); throw new OperatorException( this, "grouping fields must declare same types:" + lhsError.printVerbose() + " not same as " + rhsError.printVerbose() ); } } return true; } private Type getCanonicalType( Type type ) { if( type instanceof CoercibleType ) type = ( (CoercibleType) type ).getCanonicalType(); if( type instanceof Class ) type = Coercions.asNonPrimitive( (Class) type ); return type; } private boolean verifySameSize( Map<String, Fields> groupingFields ) { Iterator<Fields> iterator = groupingFields.values().iterator(); int size = iterator.next().size(); while( iterator.hasNext() ) { Fields groupingField = iterator.next(); if( groupingField.size() != size ) return false; size = groupingField.size(); } return true; } private Map<String, Fields> resolveSelectorsAgainstIncoming( Set<Scope> incomingScopes, Map<String, Fields> selectors, String type ) { Map<String, Fields> resolvedFields = new HashMap<String, Fields>(); for( Scope incomingScope : incomingScopes ) { Fields selector = selectors.get( incomingScope.getName() ); if( selector == null ) throw new OperatorException( this, "no " + type + " selector found for: " + incomingScope.getName() ); Fields incomingFields; if( selector.isNone() ) incomingFields = Fields.NONE; else if( selector.isAll() ) incomingFields = incomingScope.getIncomingSpliceFields(); else if( selector.isGroup() ) incomingFields = incomingScope.getOutGroupingFields(); else if( selector.isValues() ) incomingFields = incomingScope.getOutValuesFields().subtract( incomingScope.getOutGroupingFields() ); else incomingFields = incomingScope.getIncomingSpliceFields().select( selector ); resolvedFields.put( incomingScope.getName(), incomingFields ); } return resolvedFields; } Map<String, Fields> resolveSortingSelectors( Set<Scope> incomingScopes ) { try { if( getSortingSelectors().isEmpty() ) return null; return resolveSelectorsAgainstIncoming( incomingScopes, getSortingSelectors(), "sorting" ); } catch( FieldsResolverException exception ) { throw new OperatorException( this, OperatorException.Kind.sorting, exception.getSourceFields(), exception.getSelectorFields(), exception ); } catch( RuntimeException exception ) { throw new OperatorException( this, "could not resolve sorting selector in: " + this, exception ); } } @Override public Fields resolveIncomingOperationPassThroughFields( Scope incomingScope ) { return incomingScope.getIncomingSpliceFields(); } Fields resolveDeclared( Set<Scope> incomingScopes ) { try { Fields declaredFields = getJoinDeclaredFields(); if( declaredFields != null && declaredFields.isNone() ) { if( !isCoGroup() ) throw new IllegalArgumentException( "Fields.NONE may only be declared as the join fields when using a CoGroup" ); return Fields.NONE; } if( declaredFields != null ) { if( incomingScopes.size() != pipes.size() && isSelfJoin() ) throw new OperatorException( this, "self joins without intermediate operators are not permitted, see 'numSelfJoins' constructor or identity function" ); int size = 0; boolean foundUnknown = false; List<Fields> appendableFields = getOrderedResolvedFields( incomingScopes ); for( Fields fields : appendableFields ) { foundUnknown = foundUnknown || fields.isUnknown(); size += fields.size(); } if( !foundUnknown && declaredFields.size() != size * ( numSelfJoins + 1 ) ) { if( isSelfJoin() ) throw new OperatorException( this, "declared grouped fields not same size as grouped values, declared: " + declaredFields.printVerbose() + " != size: " + size * ( numSelfJoins + 1 ) ); else throw new OperatorException( this, "declared grouped fields not same size as grouped values, declared: " + declaredFields.printVerbose() + " resolved: " + Util.print( appendableFields, "" ) ); } int i = 0; for( Fields appendableField : appendableFields ) { Type[] types = appendableField.getTypes(); if( types == null ) { i += appendableField.size(); continue; } for( Type type : types ) { if( type != null ) declaredFields = declaredFields.applyType( i, type ); i++; } } return declaredFields; } if( isGroupBy() || isMerge() ) { Iterator<Scope> iterator = incomingScopes.iterator(); Fields commonFields = iterator.next().getIncomingSpliceFields(); while( iterator.hasNext() ) { Scope incomingScope = iterator.next(); Fields fields = incomingScope.getIncomingSpliceFields(); if( !commonFields.equalsFields( fields ) ) throw new OperatorException( this, "merged streams must declare the same field names, in the same order, expected: " + commonFields.printVerbose() + " found: " + fields.printVerbose() ); } return commonFields; } else { List<Fields> appendableFields = getOrderedResolvedFields( incomingScopes ); Fields appendedFields = new Fields(); try { for( Fields appendableField : appendableFields ) appendedFields = appendedFields.append( appendableField ); } catch( TupleException exception ) { String fields = ""; for( Fields appendableField : appendableFields ) fields += appendableField.print(); throw new OperatorException( this, "found duplicate field names in joined tuple stream: " + fields, exception ); } return appendedFields; } } catch( OperatorException exception ) { throw exception; } catch( RuntimeException exception ) { throw new OperatorException( this, "could not resolve declared fields in: " + this, exception ); } } public Fields getJoinDeclaredFields() { Fields declaredFields = getDeclaredFields(); if( !( joiner instanceof DeclaresResults ) ) return declaredFields; if( declaredFields == null && ( (DeclaresResults) joiner ).getFieldDeclaration() != null ) declaredFields = ( (DeclaresResults) joiner ).getFieldDeclaration(); return declaredFields; } private List<Fields> getOrderedResolvedFields( Set<Scope> incomingScopes ) { Map<String, Scope> scopesMap = new HashMap<String, Scope>(); for( Scope incomingScope : incomingScopes ) scopesMap.put( incomingScope.getName(), incomingScope ); List<Fields> appendableFields = new ArrayList<Fields>(); for( Pipe pipe : pipes ) appendableFields.add( scopesMap.get( pipe.getName() ).getIncomingSpliceFields() ); return appendableFields; } @Override @SuppressWarnings({"RedundantIfStatement"}) public boolean equals( Object object ) { if( this == object ) return true; if( object == null || getClass() != object.getClass() ) return false; if( !super.equals( object ) ) return false; Splice splice = (Splice) object; if( spliceName != null ? !spliceName.equals( splice.spliceName ) : splice.spliceName != null ) return false; if( keyFieldsMap != null ? !keyFieldsMap.equals( splice.keyFieldsMap ) : splice.keyFieldsMap != null ) return false; if( pipes != null ? !pipes.equals( splice.pipes ) : splice.pipes != null ) return false; return true; } @Override public int hashCode() { int result = super.hashCode(); result = 31 * result + ( pipes != null ? pipes.hashCode() : 0 ); result = 31 * result + ( keyFieldsMap != null ? keyFieldsMap.hashCode() : 0 ); result = 31 * result + ( spliceName != null ? spliceName.hashCode() : 0 ); return result; } @Override public String toString() { StringBuilder buffer = new StringBuilder( super.toString() ); buffer.append( "[by:" ); for( String name : keyFieldsMap.keySet() ) { if( keyFieldsMap.size() > 1 ) buffer.append( " " ).append( name ).append( ":" ); buffer.append( keyFieldsMap.get( name ).printVerbose() ); } if( isSelfJoin() ) buffer.append( "[numSelfJoins:" ).append( numSelfJoins ).append( "]" ); buffer.append( "]" ); return buffer.toString(); } @Override protected void printInternal( StringBuffer buffer, Scope scope ) { super.printInternal( buffer, scope ); Map<String, Fields> map = scope.getKeySelectors(); if( map != null ) { buffer.append( "[by:" ); for( Map.Entry<String, Fields> entry : keyFieldsMap.entrySet() ) { String name = entry.getKey(); if( map.size() > 1 ) buffer.append( name ).append( ":" ); Fields keys = map.get( name ); if( keys == null ) buffer.append( "<unavailable>" ); else buffer.append( keys.print() ); } if( isSelfJoin() ) buffer.append( "[numSelfJoins:" ).append( numSelfJoins ).append( "]" ); buffer.append( "]" ); } } }
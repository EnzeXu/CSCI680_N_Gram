public class Sorter extends Ordering implements Comparator<Description> { public static final Sorter NULL = new Sorter(new Comparator<Description>() { public int compare(Description o1, Description o2) { return 0; } }); private final Comparator<Description> comparator; public Sorter(Comparator<Description> comparator) { this.comparator = comparator; } @Override public void apply(Object target) { if (target instanceof Sortable) { Sortable sortable = (Sortable) target; sortable.sort(this); } } public int compare(Description o1, Description o2) { return comparator.compare(o1, o2); } @Override protected final List<Description> orderItems(Collection<Description> descriptions) { List<Description> sorted = new ArrayList<Description>(descriptions); Collections.sort(sorted, this); return sorted; } @Override boolean validateOrderingIsCorrect() { return false; } }
public class ConnectbotMatchers { @NonNull public static Matcher<RecyclerView.ViewHolder> withHostNickname(final String content) { return new BoundedMatcher<RecyclerView.ViewHolder, HostListActivity.HostViewHolder>(HostListActivity.HostViewHolder.class) { @Override public boolean matchesSafely(HostListActivity.HostViewHolder holder) { return holder.host.getNickname().matches(content); } @Override public void describeTo(Description description) { description.appendText("with host nickname '" + content + "'"); } }; } @NonNull public static Matcher<RecyclerView.ViewHolder> withPubkeyNickname(final String content) { return new BoundedMatcher<RecyclerView.ViewHolder, PubkeyListActivity.PubkeyViewHolder>(PubkeyListActivity.PubkeyViewHolder.class) { @Override public boolean matchesSafely(PubkeyListActivity.PubkeyViewHolder holder) { return holder.pubkey.getNickname().matches(content); } @Override public void describeTo(Description description) { description.appendText("with host nickname '" + content + "'"); } }; } @NonNull public static Matcher<RecyclerView.ViewHolder> withConnectedHost() { return new BoundedMatcher<RecyclerView.ViewHolder, HostListActivity.HostViewHolder>(HostListActivity.HostViewHolder.class) { @Override public boolean matchesSafely(HostListActivity.HostViewHolder holder) { return hasDrawableState(holder.icon, android.R.attr.state_checked); } @Override public void describeTo(Description description) { description.appendText("is connected status"); } }; } @NonNull public static Matcher<RecyclerView.ViewHolder> withDisconnectedHost() { return new BoundedMatcher<RecyclerView.ViewHolder, HostListActivity.HostViewHolder>(HostListActivity.HostViewHolder.class) { @Override public boolean matchesSafely(HostListActivity.HostViewHolder holder) { return hasDrawableState(holder.icon, android.R.attr.state_expanded); } @Override public void describeTo(Description description) { description.appendText("is disconnected status"); } }; } @NonNull public static Matcher<RecyclerView.ViewHolder> withColoredText(@ColorInt final int expectedColor) { return new BoundedMatcher<RecyclerView.ViewHolder, HostListActivity.HostViewHolder>(HostListActivity.HostViewHolder.class) { @Override public boolean matchesSafely(HostListActivity.HostViewHolder holder) { return hasTextColor(holder.nickname, expectedColor); } @Override public void describeTo(Description description) { description.appendText("is text color " + expectedColor); } }; } private static boolean hasDrawableState(View view, final int expectedState) { if (!(view instanceof ImageView)) { return false; } int[] states = view.getDrawableState(); for (int state : states) { if (state == expectedState) { return true; } } return false; } private static boolean hasTextColor(View view, @ColorInt final int expectedColor) { if (!(view instanceof TextView)) { return false; } TextView tv = (TextView) view; return tv.getCurrentTextColor() == expectedColor; } @NonNull public static ViewAssertion hasHolderItem(final Matcher<RecyclerView.ViewHolder> viewHolderMatcher) { return new ViewAssertion() { @Override public void check(View view, NoMatchingViewException e) { if (!(view instanceof RecyclerView)) { throw e; } boolean hasMatch = false; RecyclerView rv = (RecyclerView) view; for (int i = 0; i < rv.getChildCount(); i++) { RecyclerView.ViewHolder vh = rv.findViewHolderForAdapterPosition(i); hasMatch |= viewHolderMatcher.matches(vh); } Assert.assertTrue(hasMatch); } }; } }
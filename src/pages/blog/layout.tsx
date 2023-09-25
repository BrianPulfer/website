import BlogLayout from '@/components/Layout/BlogLayout'

export default function Layout ({ children }: { children: React.ReactNode }): JSX.Element {
  return <BlogLayout>{children}</BlogLayout>
}

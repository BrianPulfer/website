import BlogLayout from '@/components/Layout/BlogLayout'

export default function Layout ({ children }: { children: React.ReactNode }) {
  return <BlogLayout>{children}</BlogLayout>
}
